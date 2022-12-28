#include "Evaluation/Status.cuh"
#include "IntegrateByParts.cuh"

#include "Evaluation/StaticFunctions.cuh"
#include "Symbol/MetaOperators.cuh"
#include "Symbol/SubexpressionCandidate.cuh"
#include "Symbol/Symbol.cuh"
#include "Symbol/TreeIterator.cuh"
#include "Utils/CompileConstants.cuh"
#include "Utils/Order.cuh"

namespace Sym::Heuristic {
    namespace {
        __host__ __device__ void extract_second_factor(const Symbol& product,
                                                       const Symbol& first_factor,
                                                       Symbol& destination, Symbol& help_space) {
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
                const auto order = Symbol::compare_expressions(
                    *product_it.current(), *first_factor_it.current(), help_space);

                switch (order) {
                case Util::Order::Equal:
                    product_it.advance();
                    first_factor_it.advance();
                    break;
                case Util::Order::Greater:
                    Symbol::copy_and_reverse_symbol_sequence(*current_dst, *product_it.current(),
                                                             product_it.current()->size());
                    current_dst += current_dst->size();
                    ++expressions_copied;
                    product_it.advance();
                    break;
                case Util::Order::Less:
                    Symbol::copy_and_reverse_symbol_sequence(*current_dst,
                                                             *first_factor_it.current(),
                                                             first_factor_it.current()->size());
                    current_dst += current_dst->size();
                    ++expressions_copied;
                    first_factor_it.advance();
                    break;
                }
            }

            auto* const valid_it = product_it.is_valid() ? &product_it : &first_factor_it;
            while (valid_it->is_valid()) {
                Symbol::copy_and_reverse_symbol_sequence(*current_dst, *valid_it->current(),
                                                         valid_it->current()->size());
                current_dst += current_dst->size();
                ++expressions_copied;
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
    }

    __device__ CheckResult is_simple_function(const Integral& integral, Symbol&  /*help_space*/) {
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

        const typename ExpressionType::AdditionalArgs EXPRESSION_ARGS = {
            {integral.vacancy_expression_idx, integral.vacancy_idx, 1},
            first_function,
            second_factor_size,
        };

        ENSURE_ENOUGH_SPACE(ExpressionType::size_with(EXPRESSION_ARGS), expression_dst);

        ExpressionType::init(*expression_dst, EXPRESSION_ARGS);

        Symbol& second_factor_dst = (*expression_dst)
                                        .as<SubexpressionCandidate>()
                                        .arg()
                                        .as<Addition>()
                                        .arg2()
                                        .as<Product>()
                                        .arg2();

        ENSURE_ENOUGH_SPACE(integrand.size(), help_space);

        extract_second_factor(integrand, first_function_derivative, second_factor_dst, *help_space);

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

        const typename IntegralType::AdditionalArgs INTEGRAL_ARGS = {
            {expression_dst.index(), 3, 0},
            original_integral,
            first_function,
            *help_iterator,
        };

        ENSURE_ENOUGH_SPACE(IntegralType::size_with(INTEGRAL_ARGS), integral_dst);

        IntegralType::init(*integral_dst, INTEGRAL_ARGS);

        return EvaluationStatus::Done;
    }
}