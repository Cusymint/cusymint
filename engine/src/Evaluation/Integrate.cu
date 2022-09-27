#include "Integrate.cuh"

#include "Utils/Cuda.cuh"

namespace {
    __device__ bool is_zero_size(const size_t index,
                                 const Util::DeviceArray<size_t>& inclusive_scan) {
        return index == 0 && inclusive_scan[index] != 0 ||
               index != 0 && inclusive_scan[index - 1] != inclusive_scan[index];
    }
}

namespace Sym {
    __device__ const ApplicabilityCheck known_integral_checks[] = {
        is_single_variable, is_simple_variable_power, is_variable_exponent,
        is_simple_sine,     is_simple_cosine,         is_constant,
        is_known_arctan};

    __device__ const IntegralTransform known_integral_applications[] = {
        integrate_single_variable, integrate_simple_variable_power, integrate_variable_exponent,
        integrate_simple_sine,     integrate_simple_cosine,         integrate_constant,
        integrate_arctan};

    static_assert(sizeof(known_integral_applications) == sizeof(known_integral_checks),
                  "Different number of heuristics and applications defined");

    static_assert(sizeof(known_integral_checks) ==
                      sizeof(ApplicabilityCheck) * KNOWN_INTEGRAL_COUNT,
                  "HEURISTIC_CHECK_COUNT is not equal to number of heuristic checks");

    __device__ const ApplicabilityCheck heuristic_checks[] = {is_function_of_ex};

    __device__ const IntegralTransform heuristic_applications[] = {transform_function_of_ex};

    static_assert(sizeof(heuristic_checks) == sizeof(heuristic_applications),
                  "Different number of heuristics and applications defined");

    static_assert(sizeof(heuristic_checks) == sizeof(ApplicabilityCheck) * HEURISTIC_CHECK_COUNT,
                  "HEURISTIC_CHECK_COUNT is not equal to number of heuristic checks");

    __device__ Symbol ex_function[3];
    __device__ void init_ex_function() {
        Power* const power = ex_function << Power::builder();
        power->arg1().known_constant = KnownConstant::with_value(KnownConstantValue::E);
        power->seal_arg1();
        power->arg2().variable = Variable::create();
        power->seal();
    }

    __device__ size_t is_function_of_ex(const Integral* const integral) {
        // TODO: Move somewhere so that it's initialized only once and not every time this function
        // is called
        init_ex_function();
        return integral->integrand()->is_function_of(ex_function) ? 1 : 0;
    }

    __device__ void transform_function_of_ex(const Integral* const integral,
                                             Symbol* const destination, Symbol* const help_space) {
        // TODO: Move somewhere so that it's initialized only once and not every time this function
        // is called
        init_ex_function();
        Symbol variable{};
        variable.variable = Variable::create();

        integral->integrate_by_substitution_with_derivative(ex_function, &variable, destination,
                                                            help_space);
    }

    __device__ size_t is_single_variable(const Integral* const integral) {
        return integral->integrand()->is(Type::Variable) ? 1 : 0;
    }

    __device__ size_t is_simple_variable_power(const Integral* const integral) {
        const Symbol* const integrand = integral->integrand();
        if (!integrand[0].is(Type::Power) || !integrand[1].is(Type::Variable)) {
            return 0;
        }

        if (integrand[2].is(Type::NumericConstant) && integrand[2].numeric_constant.value == -1.0) {
            return 0;
        }

        return integrand[2].is_constant() ? 0 : 1;
    }
    __device__ size_t is_variable_exponent(const Integral* const integral) {
        const Symbol* const integrand = integral->integrand();
        return integrand[0].is(Type::Power) && integrand[1].is(Type::KnownConstant) &&
                       integrand[1].known_constant.value == KnownConstantValue::E &&
                       integrand[2].is(Type::Variable)
                   ? 1
                   : 0;
    }
    __device__ size_t is_simple_sine(const Integral* const integral) {
        const Symbol* const integrand = integral->integrand();
        return integrand[0].is(Type::Sine) && integrand[1].is(Type::Variable) ? 1 : 0;
    }

    __device__ size_t is_simple_cosine(const Integral* const integral) {
        const Symbol* const integrand = integral->integrand();
        return integrand[0].is(Type::Cosine) && integrand[1].is(Type::Variable) ? 1 : 0;
    }

    __device__ size_t is_constant(const Integral* const integral) {
        const Symbol* const integrand = integral->integrand();
        return integrand->is_constant() ? 1 : 0;
    }

    __device__ size_t is_known_arctan(const Integral* const integral) {
        const Symbol* const integrand = integral->integrand();
        // 1/(x^2+1) or 1/(1+x^2)
        return integrand[0].is(Type::Product) && integrand[1].is(Type::NumericConstant) &&
                       integrand[1].numeric_constant.value == 1.0 &&
                       integrand[2].is(Type::Reciprocal) && integrand[3].is(Type::Addition) &&
                       ((integrand[4].is(Type::Power) && integrand[5].is(Type::Variable) &&
                         integrand[6].is(Type::NumericConstant) &&
                         integrand[6].numeric_constant.value == 2.0 &&
                         integrand[7].is(Type::NumericConstant) &&
                         integrand[7].numeric_constant.value == 1.0) ||
                        (integrand[4].is(Type::NumericConstant) &&
                         integrand[4].numeric_constant.value == 1.0 &&
                         integrand[5].is(Type::Power) && integrand[6].is(Type::Variable) &&
                         integrand[7].is(Type::NumericConstant) &&
                         integrand[7].numeric_constant.value == 2.0))
                   ? 1
                   : 0;
    }

    __device__ void integrate_single_variable(const Integral* const integral,
                                              Symbol* const destination,
                                              Symbol* const /*help_space*/) {

        Symbol* const solution_expr = prepare_solution(integral, destination);

        Product* const product = solution_expr << Product::builder();
        product->arg1().numeric_constant = NumericConstant::with_value(0.5);
        product->seal_arg1();

        Power* const power = product->arg2() << Power::builder();
        power->arg1().variable = Variable::create();
        power->seal_arg1();
        power->arg2().numeric_constant = NumericConstant::with_value(2.0);
        power->seal();
        product->seal();

        destination->solution.seal();
    }

    __device__ void integrate_simple_variable_power(const Integral* const integral,
                                                    Symbol* const destination,
                                                    Symbol* const /*help_space*/) {
        const Symbol* const integrand = integral->integrand();

        Symbol* const solution_expr = prepare_solution(integral, destination);
        const Symbol* const exponent = &integral->integrand()->power.arg2();

        // 1/(c+1) * x^(c+1), c może być całym drzewem
        Product* const product = solution_expr << Product::builder();

        Reciprocal* const reciprocal = product->arg1() << Reciprocal::builder();
        Addition* const multiplier_addition = reciprocal->arg() << Addition::builder();
        exponent->copy_to(&multiplier_addition->arg1());
        multiplier_addition->seal_arg1();
        multiplier_addition->arg2().numeric_constant = NumericConstant::with_value(1.0);
        multiplier_addition->seal();
        reciprocal->seal();
        product->seal_arg1();

        Power* const power = product->arg2() << Power::builder();
        power->arg1().variable = Variable::create();
        power->seal_arg1();
        Addition* const exponent_addition = power->arg2() << Addition::builder();
        exponent->copy_to(&exponent_addition->arg1());
        exponent_addition->seal_arg1();
        exponent_addition->arg2().numeric_constant = NumericConstant::with_value(1.0);
        exponent_addition->seal();
        power->seal();
        product->seal();

        destination->solution.seal();
    }

    __device__ void integrate_variable_exponent(const Integral* const integral,
                                                Symbol* const destination,
                                                Symbol* const /*help_space*/) {
        Symbol* const solution_expr = prepare_solution(integral, destination);
        const Symbol* const integrand = integral->integrand();

        Power* const power = solution_expr << Power::builder();
        power->arg1().known_constant = KnownConstant::with_value(KnownConstantValue::E);
        power->seal_arg1();
        power->arg2().variable = Variable::create();
        power->seal();

        destination->solution.seal();
    }

    __device__ void integrate_simple_sine(const Integral* const integral, Symbol* const destination,
                                          Symbol* const /*help_space*/) {
        Symbol* const solution_expr = prepare_solution(integral, destination);
        const Symbol* const integrand = integral->integrand();

        Negation* const minus = solution_expr << Negation::builder();
        Cosine* const cos = minus->arg() << Cosine::builder();
        cos->arg().variable = Variable::create();
        cos->seal();
        minus->seal();

        destination->solution.seal();
    }

    __device__ void integrate_simple_cosine(const Integral* const integral,
                                            Symbol* const destination,
                                            Symbol* const /*help_space*/) {
        Symbol* const solution_expr = prepare_solution(integral, destination);
        const Symbol* const integrand = integral->integrand();

        Sine* const sine = solution_expr << Sine::builder();
        sine->arg().variable = Variable::create();
        sine->seal();

        destination->solution.seal();
    }

    __device__ void integrate_constant(const Integral* const integral, Symbol* const destination,
                                       Symbol* const /*help_space*/) {
        const Symbol* const integrand = integral->integrand();
        Symbol* const solution_expr = prepare_solution(integral, destination);

        Product* const product = solution_expr << Product::builder();
        product->arg1().variable = Variable::create();
        product->seal_arg1();
        integrand->copy_to(&product->arg2());
        product->seal();

        destination->solution.seal();
    }

    __device__ void integrate_arctan(const Integral* const integral, Symbol* const destination,
                                     Symbol* const /*help_space*/) {
        const Symbol* const integrand = integral->integrand();
        Symbol* const solution_expr = prepare_solution(integral, destination);

        Arctangent* const arctangent = solution_expr << Arctangent::builder();
        arctangent->arg().variable = Variable::create();
        arctangent->seal();

        destination->solution.seal();
    }

    __device__ Symbol* prepare_solution(const Integral* const integral, Symbol* const destination) {
        Solution* const solution = destination << Solution::builder();
        Symbol::copy_symbol_sequence(Symbol::from(solution->first_substitution()),
                                     Symbol::from(integral->first_substitution()),
                                     integral->substitutions_size());
        solution->seal_substitutions(integral->substitution_count, integral->substitutions_size());

        return solution->expression();
    }

    __device__ void check_applicability(const ExpressionArray<Integral>& integrals,
                                        Util::DeviceArray<size_t>& applicability,
                                        const ApplicabilityCheck* const checks,
                                        const size_t check_count) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        const size_t check_step = thread_count / TRANSFORM_GROUP_SIZE;

        for (size_t check_idx = thread_idx / TRANSFORM_GROUP_SIZE; check_idx < check_count;
             check_idx += check_step) {
            for (size_t int_idx = thread_idx % TRANSFORM_GROUP_SIZE; int_idx < integrals.size();
                 int_idx += TRANSFORM_GROUP_SIZE) {
                size_t appl_idx = MAX_INTEGRAL_COUNT * check_idx + int_idx;
                applicability[appl_idx] = checks[check_idx](integrals[int_idx]);
            }
        }
    }

    __device__ void
    apply_transforms(const ExpressionArray<Integral>& integrals, ExpressionArray<>& destinations,
                     ExpressionArray<>& help_spaces, const Util::DeviceArray<size_t>& applicability,
                     const IntegralTransform* const transforms, const size_t transform_count) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        const size_t trans_step = thread_count / TRANSFORM_GROUP_SIZE;

        for (size_t trans_idx = thread_idx / TRANSFORM_GROUP_SIZE; trans_idx < transform_count;
             trans_idx += trans_step) {
            for (size_t int_idx = thread_idx % TRANSFORM_GROUP_SIZE; int_idx < integrals.size();
                 int_idx += TRANSFORM_GROUP_SIZE) {
                const size_t appl_index = MAX_INTEGRAL_COUNT * trans_idx + int_idx;
                if (is_zero_size(appl_index, applicability)) {
                    const size_t dest_idx = applicability[appl_index] - 1;
                    transforms[trans_idx](integrals[int_idx], destinations[dest_idx],
                                          help_spaces[dest_idx]);
                }
            }
        }
    }

    __global__ void check_for_known_integrals(const ExpressionArray<Integral> integrals,
                                              Util::DeviceArray<size_t> applicability) {
        check_applicability(integrals, applicability, known_integral_checks, KNOWN_INTEGRAL_COUNT);
    }

    __global__ void apply_known_integrals(const ExpressionArray<Integral> integrals,
                                          ExpressionArray<> destinations,
                                          ExpressionArray<> help_spaces,
                                          const Util::DeviceArray<size_t> applicability) {
        apply_transforms(integrals, destinations, help_spaces, applicability,
                         known_integral_applications, KNOWN_INTEGRAL_COUNT);
    }

    __global__ void check_heuristics_applicability(const ExpressionArray<Integral> integrals,
                                                   Util::DeviceArray<size_t> applicability) {
        check_applicability(integrals, applicability, heuristic_checks, HEURISTIC_CHECK_COUNT);
    }

    __global__ void apply_heuristics(const ExpressionArray<Integral> integrals,
                                     ExpressionArray<> destinations, ExpressionArray<> help_spaces,
                                     const Util::DeviceArray<size_t> applicability) {
        apply_transforms(integrals, destinations, help_spaces, applicability,
                         heuristic_applications, HEURISTIC_CHECK_COUNT);
    }

    __global__ void simplify(ExpressionArray<> expressions, ExpressionArray<> help_spaces) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        for (size_t expr_idx = thread_idx; expr_idx < expressions.size();
             expr_idx += thread_count) {
            expressions[expr_idx]->simplify(help_spaces[expr_idx]);
        }
    }
}
