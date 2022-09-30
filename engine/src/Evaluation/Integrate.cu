#include "Integrate.cuh"

#include "Utils/Cuda.cuh"

namespace {
    __device__ bool is_zero_size(const size_t index,
                                 const Util::DeviceArray<size_t>& inclusive_scan) {
        return index == 0 && inclusive_scan[index] != 0 ||
               index != 0 && inclusive_scan[index - 1] != inclusive_scan[index];
    }

    /*
     * @brief Podejmuje próbę ustawienia `expressions[potential_solver_idx]` (będącego
     * SubexpressionCandidate) jako rozwiązania wskazywanego przez siebie SubexpressionVacancy
     *
     * @param expressions Talbica wyrażeń z kandydatem do rozwiązania i brakującym podwyrażeniem
     * @param potential_solver_idx Indeks kandydata do rozwiązania
     *
     * @return `false` jeśli nie udało się ustawić wybranego kandydata jako rozwiązanie
     * podwyrażenia, lub udało się, ale w nadwyrażeniu są jeszcze inne nierozwiązane podwyrażenia.
     * `true` jeśli się udało i było to ostatnie nierozwiązane podwyrażenie w nadwyrażeniu.
     */
    __device__ bool try_set_solver_idx(Sym::ExpressionArray<>& expressions,
                                       const size_t potential_solver_idx) {
        const size_t& vacancy_expr_idx =
            expressions[potential_solver_idx]->subexpression_candidate.vacancy_expression_idx;

        const size_t& vacancy_idx =
            expressions[potential_solver_idx]->subexpression_candidate.vacancy_idx;

        Sym::SubexpressionVacancy& subexpr_vacancy =
            expressions[vacancy_expr_idx][vacancy_idx].subexpression_vacancy;

        const bool solver_lock_acquired = atomicCAS(&subexpr_vacancy.is_solved, 0, 1) == 0;

        if (!solver_lock_acquired) {
            return false;
        }

        subexpr_vacancy.solver_idx = potential_solver_idx;

        if (!expressions[vacancy_expr_idx]->is(Sym::Type::SubexpressionCandidate)) {
            return true;
        }

        unsigned int subexpressions_left = atomicSub(
            &expressions[vacancy_expr_idx]->subexpression_candidate.subexpressions_left, 1);

        return subexpressions_left == 0;
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
                size_t appl_idx = MAX_EXPRESSION_COUNT * check_idx + int_idx;
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
                const size_t appl_index = MAX_EXPRESSION_COUNT * trans_idx + int_idx;
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

    __global__ void apply_known_integrals(const ExpressionArray<SubexpressionCandidate> integrals,
                                          ExpressionArray<> expressions,
                                          ExpressionArray<> help_spaces,
                                          const Util::DeviceArray<size_t> applicability) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        const size_t trans_step = thread_count / TRANSFORM_GROUP_SIZE;

        for (size_t trans_idx = thread_idx / TRANSFORM_GROUP_SIZE; trans_idx < KNOWN_INTEGRAL_COUNT;
             trans_idx += trans_step) {
            for (size_t int_idx = thread_idx % TRANSFORM_GROUP_SIZE; int_idx < integrals.size();
                 int_idx += TRANSFORM_GROUP_SIZE) {
                const size_t appl_index = MAX_EXPRESSION_COUNT * trans_idx + int_idx;

                if (!is_zero_size(appl_index, applicability)) {
                    continue;
                }

                const size_t dest_idx = expressions.size() + applicability[appl_index] - 1;

                auto* const subexpr_candidate = expressions[dest_idx]
                                                << SubexpressionCandidate::builder();
                subexpr_candidate->copy_metadata_from(*integrals[int_idx]);
                known_integral_applications[trans_idx](&integrals[int_idx]->arg().integral,
                                                       &subexpr_candidate->arg(),
                                                       help_spaces[dest_idx]);
                subexpr_candidate->seal();

                try_set_solver_idx(expressions, dest_idx);
            }
        }
    }

    __global__ void propagate_solved_subexpressions(ExpressionArray<> expressions) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        // W każdym węźle drzewa zależności wyrażeń zaczyna jeden wątek. Jeśli jego węzeł jest
        // rozwiązany, to próbuje się ustawić jako rozwiązanie swojego podwyrażenia w rodzicu. Jeśli
        // mu się to uda i nie pozostaną w rodzicu inne nierozwiązane podwyrażenia, to przechodzi do
        // niego i powtaża wszystko. W skrócie następuje propagacja informacji o rozwiązaniu z dołu
        // drzewa na samą górę.

        // Na expr_idx = 0 jest tylko SubexpressionVacancy oryginalnej całki, więc pomijamy
        for (size_t expr_idx = thread_idx + 1; expr_idx < expressions.size();
             expr_idx += thread_count) {
            while (expr_idx != 0) {
                if (expressions[expr_idx]->subexpression_candidate.subexpressions_left != 0) {
                    break;
                }

                if (!try_set_solver_idx(expressions, expr_idx)) {
                    break;
                }

                // Przechodzimy w drzewie zależności do rodzica. Być może będziemy tam razem z
                // wątkiem, który tam zaczął pętlę. `try_set_solver_idx` jest jednak atomowe, więc
                // tylko jednemu z wątków uda się ustawić `solver_idx` na kolejnym rodzicu, więc
                // tylko jeden wątek tam przetrwa.
                expr_idx = expressions[expr_idx]->subexpression_candidate.vacancy_expression_idx;
            }
        }
    }

    __global__ void find_redundand_expressions(const ExpressionArray<> expressions,
                                               Util::DeviceArray<size_t> removability) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        // Szukamy coraz wyżej w drzewie zależności, czy nie próbujemy rozwiązać czegoś, co zostało
        // już rozwiązane w inny sposób.

        // Na expr_idx = 0 jest tylko SubexpressionVacancy oryginalnej całki, więc pomijamy
        for (size_t expr_idx = thread_idx + 1; expr_idx < expressions.size();
             expr_idx += thread_count) {
            size_t current_expr_idx = 0;
            removability[expr_idx] = 1;

            while (current_expr_idx != 0) {
                const size_t& parent_idx =
                    expressions[current_expr_idx]->subexpression_candidate.vacancy_expression_idx;
                const size_t& parent_vacancy_idx =
                    expressions[current_expr_idx]->subexpression_candidate.vacancy_idx;
                const SubexpressionVacancy& parent_vacancy =
                    expressions[parent_idx][parent_vacancy_idx].subexpression_vacancy;

                if (parent_vacancy.is_solved == 1 &&
                    parent_vacancy.solver_idx != current_expr_idx) {
                    removability[expr_idx] = 0;
                    break;
                }

                current_expr_idx = parent_idx;
            }
        }
    }

    __global__ void
    find_redundand_integrals(const ExpressionArray<> integrals, const ExpressionArray<> expressions,
                             const Util::DeviceArray<size_t> expressions_removability,
                             Util::DeviceArray<size_t> integrals_removability) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        for (size_t int_idx = thread_idx + 1; int_idx < integrals.size(); int_idx += thread_count) {
            const size_t& vacancy_expr_idx =
                integrals[int_idx]->subexpression_candidate.vacancy_expression_idx;
            const size_t& vacancy_idx = integrals[int_idx]->subexpression_candidate.vacancy_idx;

            bool parent_expr_failed = expressions_removability[vacancy_expr_idx] == 0;
            bool parent_vacancy_solved =
                expressions[vacancy_expr_idx][vacancy_idx].subexpression_vacancy.is_solved == 1;

            integrals_removability[int_idx] = parent_expr_failed || parent_vacancy_solved ? 0 : 1;
        }
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

    __global__ void did_integrals_expire(const ExpressionArray<> expressions,
                                         const Util::DeviceArray<size_t> expressions_removability,
                                         const ExpressionArray<SubexpressionCandidate> integrals,
                                         Util::DeviceArray<size_t> integrals_removability) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        for (size_t int_idx = 0; int_idx < integrals.size(); int_idx += thread_count) {
            const size_t expr_idx = integrals[int_idx]->vacancy_expression_idx;
            const size_t subexpr_idx = integrals[int_idx]->vacancy_idx;

            const bool will_expression_be_removed = expressions_removability[expr_idx] == 0;
            const bool is_subexpression_solved =
                expressions[expr_idx][subexpr_idx].subexpression_vacancy.is_solved;

            integrals_removability[int_idx] =
                is_subexpression_solved || will_expression_be_removed ? 0 : 1;
        }
    }

    __global__ void are_integrals_failed(const Util::DeviceArray<size_t> expressions_removability,
                                         const ExpressionArray<SubexpressionCandidate> integrals,
                                         Util::DeviceArray<size_t> integrals_removability) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        for (size_t int_idx = 0; int_idx < integrals.size(); int_idx += thread_count) {
            const size_t expr_idx = integrals[int_idx]->vacancy_expression_idx;
            integrals_removability[int_idx] = expressions_removability[expr_idx];
        }
    }

    __global__ void remove_expressions(const ExpressionArray<> expressions,
                                       const Util::DeviceArray<size_t> removability,
                                       ExpressionArray<> destinations) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        for (size_t expr_idx = 0; expr_idx < expressions.size(); expr_idx += thread_count) {
            if (is_zero_size(expr_idx, removability)) {
                Symbol* destination = destinations[removability[expr_idx] - 1];
                expressions[expr_idx]->copy_to(destination);

                if (destination->is(Type::SubexpressionCandidate)) {
                    size_t& vacancy_expr_idx =
                        destination->subexpression_candidate.vacancy_expression_idx;
                    vacancy_expr_idx = removability[vacancy_expr_idx] - 1;
                }

                for (size_t symbol_idx = 0; symbol_idx < destination->size(); ++symbol_idx) {
                    if (destination[symbol_idx].is(Type::SubexpressionVacancy) &&
                        destination->subexpression_vacancy.is_solved) {
                        size_t& solver_idx = destination->subexpression_vacancy.solver_idx;
                        solver_idx = removability[solver_idx] - 1;
                    }
                }
            }
        }
    }

    __global__ void
    remove_expressions(ExpressionArray<> expressions, const Util::DeviceArray<size_t> removability,
                       ExpressionArray<SubexpressionCandidate> subexpression_candidates,
                       ExpressionArray<> destinations) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        for (size_t expr_idx = thread_idx; expr_idx < expressions.size();
             expr_idx += thread_count) {
            if (is_zero_size(expr_idx, removability)) {
                const size_t destination_idx = removability[expr_idx] - 1;
                expressions[expr_idx]->copy_to(destinations[destination_idx]);
            }
        }

        for (size_t candidate_idx = thread_idx; candidate_idx < subexpression_candidates.size();
             candidate_idx += thread_count) {
            const size_t expr_idx = subexpression_candidates[candidate_idx]->vacancy_expression_idx;
            subexpression_candidates[candidate_idx]->vacancy_expression_idx =
                removability[expr_idx] - 1;
        }
    }

    __global__ void remove_integrals(const ExpressionArray<SubexpressionCandidate> integrals,
                                     const Util::DeviceArray<size_t> removability,
                                     ExpressionArray<> destinations) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        for (size_t int_idx = thread_idx; int_idx < integrals.size(); int_idx += thread_count) {
            if (is_zero_size(int_idx, removability)) {
            }
        }
    }

    __global__ void zero_candidate_integral_count(ExpressionArray<> expressions) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        for (size_t expr_idx = thread_idx; expr_idx < expressions.size();
             expr_idx += thread_count) {
            for (size_t symbol_idx = 0; symbol_idx < expressions[expr_idx]->size(); ++symbol_idx) {
                expressions[expr_idx][symbol_idx].if_is_do<SubexpressionVacancy>(
                    [](const auto sym) { sym->candidate_integral_count = 0; });
            }
        }
    }
}
