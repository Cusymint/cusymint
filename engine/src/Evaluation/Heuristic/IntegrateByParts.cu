#include "Evaluation/Status.cuh"
#include "IntegrateByParts.cuh"

#include "Evaluation/StaticFunctions.cuh"
#include "Symbol/Constants.cuh"
#include "Symbol/MetaOperators.cuh"
#include "Symbol/SubexpressionCandidate.cuh"
#include "Symbol/Symbol.cuh"
#include "Symbol/TreeIterator.cuh"
#include "Utils/CompileConstants.cuh"
#include "Utils/Cuda.cuh"
#include "Utils/Order.cuh"

namespace Sym::Heuristic {
    namespace {
        __host__ __device__ void extract_second_factor(const Symbol& product,
                                                       const Symbol& first_factor,
                                                       Symbol& destination) {
            if (first_factor.is(1)) {
                product.copy_to(destination);
                return;
            }

            // assume that first_factor and product are sorted
            ConstTreeIterator<Product> product_it(&product);
            ConstTreeIterator<Product> first_factor_it(&first_factor);

            Symbol* current_dst = &destination;

            size_t expressions_copied = 0;

            while (product_it.is_valid() && first_factor_it.is_valid()) {
                if (!Symbol::are_expressions_equal(*product_it.current(),
                                                   *first_factor_it.current())) {
                    Symbol::copy_and_reverse_symbol_sequence(*current_dst, *product_it.current(),
                                                             product_it.current()->size());
                    current_dst += product_it.current()->size();
                    ++expressions_copied;
                }
                else {
                    first_factor_it.advance();
                }
                product_it.advance();
            }

            while (product_it.is_valid()) {
                Symbol::copy_and_reverse_symbol_sequence(*current_dst, *product_it.current(),
                                                         product_it.current()->size());
                current_dst += product_it.current()->size();
                ++expressions_copied;
                product_it.advance();
            }

            if (expressions_copied == 0) {
                destination.init_from(NumericConstant::with_value(1));
                return;
            }
            for (int i = 0; i < expressions_copied - 1; ++i) {
                Mul<None, None>::init_reverse(current_dst[i]);
            }
            const size_t size = (current_dst - &destination) + expressions_copied - 1;
            Symbol::reverse_symbol_sequence(&destination, size);
        }

        __host__ __device__ bool
        is_derivative_going_to_simplify_expression(const Symbol& expression) {
            for (size_t i = 0; i < expression.size(); ++i) {
                const Symbol& current = *expression.at(i);
                switch (current.type()) {
                case Type::Sine:
                case Type::Cosine:
                case Type::Tangent:
                case Type::Cotangent:
                    return false;
                case Type::Power: {
                    const auto& second_arg = current.as<Power>().arg2();
                    if (!second_arg.is(Type::NumericConstant) ||
                        second_arg.as<NumericConstant>().value < 0) {
                        return false;
                    }
                } break;
                default:
                    break;
                }
            }
            return true;
        }
    }

    __device__ CheckResult is_simple_function(const Integral& integral, Symbol& /*help_space*/) {
        if (integral.integrand().size() == 2) {
            return {1, 1};
        }

        return {0, 0};
    }

    __device__ EvaluationStatus integrate_simple_function_by_parts(
        const SubexpressionCandidate& integral, const ExpressionArray<>::Iterator& integral_dst,
        const ExpressionArray<>::Iterator& expression_dst,
        const ExpressionArray<>::Iterator& help_space) {
        return integrate_by_parts(integral, Static::one(), Static::identity(), integral_dst,
                                  expression_dst, help_space);
    }

    __device__ CheckResult is_product_with_exponential(const Integral& integral,
                                                       Symbol& help_space) {
        return is_product_with(Static::e_to_x(), integral, help_space);
    }

    __device__ EvaluationStatus integrate_exp_product_by_parts(
        const SubexpressionCandidate& integral, const ExpressionArray<>::Iterator& integral_dst,
        const ExpressionArray<>::Iterator& expression_dst,
        const ExpressionArray<>::Iterator& help_space) {
        return integrate_by_parts(integral, Static::e_to_x(), Static::e_to_x(), integral_dst,
                                  expression_dst, help_space);
    }

    __device__ CheckResult is_product_with_sine(const Integral& integral, Symbol& help_space) {
        return is_product_with(Static::sin_x(), integral, help_space);
    }
    __device__ EvaluationStatus integrate_sine_product_by_parts(
        const SubexpressionCandidate& integral, const ExpressionArray<>::Iterator& integral_dst,
        const ExpressionArray<>::Iterator& expression_dst,
        const ExpressionArray<>::Iterator& help_space) {
        return integrate_by_parts(integral, Static::sin_x(), Static::neg_cos_x(), integral_dst,
                                  expression_dst, help_space);
    }

    __device__ CheckResult is_product_with_cosine(const Integral& integral, Symbol& help_space) {
        return is_product_with(Static::cos_x(), integral, help_space);
    }
    __device__ EvaluationStatus integrate_cosine_product_by_parts(
        const SubexpressionCandidate& integral, const ExpressionArray<>::Iterator& integral_dst,
        const ExpressionArray<>::Iterator& expression_dst,
        const ExpressionArray<>::Iterator& help_space) {
        return integrate_by_parts(integral, Static::cos_x(), Static::sin_x(), integral_dst,
                                  expression_dst, help_space);
    }

    __device__ CheckResult is_product_with_power(const Integral& integral, Symbol& /*help_space*/) {
        const auto& integrand = integral.integrand();
        if (!integrand.is(Type::Product)) {
            return {0, 0};
        }

        ConstTreeIterator<Product> iterator(integrand.as_ptr<Product>());
        bool found_expression = false;

        while (iterator.is_valid()) {
            if (!found_expression && iterator.current()->is(Type::Variable)) {
                found_expression = true;
            }
            else if (!found_expression && iterator.current()->is(Type::Power)) {
                const auto& power = iterator.current()->as<Power>();
                if (power.arg1().is(Type::Variable) && power.arg2().is(Type::NumericConstant) &&
                    !power.arg2().is(-1)) {
                    found_expression = true;
                }
            }
            else if (!is_derivative_going_to_simplify_expression(*iterator.current())) {
                return {0, 0};
            }
            iterator.advance();
        }
        if (found_expression) {
            return {1, 1};
        }
        return {0, 0};
    }

    __device__ EvaluationStatus integrate_power_product_by_parts(
        const SubexpressionCandidate& integral, const ExpressionArray<>::Iterator& integral_dst,
        const ExpressionArray<>::Iterator& expression_dst,
        const ExpressionArray<>::Iterator& help_space) {
        const auto& integrand = integral.arg().as<Integral>().integrand();

        ConstTreeIterator<Product> iterator(integrand.as_ptr<Product>());

        double exponent;
        const Symbol* power_symbol = nullptr;

        while (iterator.is_valid()) {
            if (iterator.current()->is(Type::Variable)) {
                power_symbol = iterator.current();
                exponent = 1;
                break;
            }
            if (iterator.current()->is(Type::Power)) {
                const auto& power = iterator.current()->as<Power>();
                if (power.arg1().is(Type::Variable) && power.arg2().is(Type::NumericConstant) &&
                    !power.arg2().is(-1)) {
                    power_symbol = iterator.current();
                    exponent = power.arg2().as<NumericConstant>().value;
                    break;
                }
            }
            iterator.advance();
        }

        if constexpr (Consts::DEBUG) {
            if (power_symbol == nullptr) {
                Util::crash("Couldn't find power factor in product");
            }
        }

        using PowerAntiDerivativeType = Mul<Num, Pow<Var, Num>>;

        ENSURE_ENOUGH_SPACE(PowerAntiDerivativeType::Size::get_value(), help_space);

        Symbol& antiderivative_dst = *help_space;

        PowerAntiDerivativeType::init(antiderivative_dst, {1 / (exponent + 1), exponent + 1});

        return integrate_by_parts(integral, *power_symbol, antiderivative_dst, integral_dst,
                                  expression_dst, help_space);
    }

    __device__ CheckResult is_product_with(const Symbol& expression, const Integral& integral,
                                           Symbol& help_space) {
        const auto& integrand = integral.integrand();
        if (!integrand.is(Type::Product) || integrand.is_function_of(&help_space, expression)) {
            return {0, 0};
        }

        ConstTreeIterator<Product> iterator(integrand.as_ptr<Product>());
        bool found_expression = false;

        while (iterator.is_valid()) {
            if (!found_expression &&
                Symbol::are_expressions_equal(*iterator.current(), expression)) {
                found_expression = true;
            }
            else if (!is_derivative_going_to_simplify_expression(*iterator.current())) {
                return {0, 0};
            }
            iterator.advance();
        }
        if (found_expression) {
            return {1, 1};
        }
        return {0, 0};
    }

    __device__ EvaluationStatus integrate_by_parts(
        const SubexpressionCandidate& integral, const Symbol& first_function_derivative,
        const Symbol& first_function, const ExpressionArray<>::Iterator& integral_dst,
        const ExpressionArray<>::Iterator& expression_dst,
        const ExpressionArray<>::Iterator& help_space) {
        const auto& integrand = integral.arg().as<Integral>().integrand();

        using ExpressionType = Candidate<Add<Neg<SingleIntegralVacancy>, Mul<Copy, Skip>>>;
        using IntegralType = Candidate<Int<Mul<Copy, Copy>>>;

        size_t second_factor_size;
        if (first_function_derivative.is(1)) {
            second_factor_size = integrand.size();
        }
        else if (integrand.size() == first_function_derivative.size()) {
            // first_function_derivative is a subexpression of integrand, so this must mean that
            // they are equal and second factor is Integer<1>
            second_factor_size = 1;
        }
        else {
            // if sizes are not equal and derivative is not 1, then integrand is a product
            second_factor_size = integrand.size() - first_function_derivative.size() - 1;
        }

        const typename ExpressionType::AdditionalArgs expression_args = {
            {integral.vacancy_expression_idx, integral.vacancy_idx, 1},
            first_function,
            second_factor_size,
        };

        ENSURE_ENOUGH_SPACE(ExpressionType::size_with(expression_args), expression_dst);

        ExpressionType::init(*expression_dst, expression_args);

        auto& second_factor_product = (*expression_dst)
                                          .as<SubexpressionCandidate>()
                                          .arg()
                                          .as<Addition>()
                                          .arg2()
                                          .as<Product>();

        Symbol& second_factor_dst = second_factor_product.arg2();
        const Symbol& first_function_copy = second_factor_product.arg1();

        extract_second_factor(integrand, first_function_derivative, second_factor_dst);

        if constexpr (Consts::DEBUG) {
            if (second_factor_dst.size() != second_factor_size) {
                Util::crash("Second factor size (%lu) does not match predicted size (%lu)",
                            second_factor_dst.size(), second_factor_size);
            }

            if (second_factor_dst.is(Type::Unknown)) {
                Util::crash("Second factor type is (still) Unknown");
            }
        }

        SymbolIterator help_iterator =
            TRY_EVALUATE_RESULT(SymbolIterator::from_at(*help_space, 0, help_space.capacity()));
        TRY_EVALUATE_RESULT(second_factor_dst.derivative_to(help_iterator));

        const auto& original_integral = integral.arg().as<Integral>();

        const typename IntegralType::AdditionalArgs integral_args = {
            {expression_dst.index(), 4, 0},
            original_integral,
            first_function_copy,
            *help_iterator,
        };

        ENSURE_ENOUGH_SPACE(IntegralType::size_with(integral_args), integral_dst);

        IntegralType::init(*integral_dst, integral_args);

        return EvaluationStatus::Done;
    }
}
