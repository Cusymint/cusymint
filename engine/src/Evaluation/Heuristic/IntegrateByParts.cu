#include "IntegrateByParts.cuh"

#include "Evaluation/StaticFunctions.cuh"
#include "Symbol/MetaOperators.cuh"
#include "Symbol/SubexpressionCandidate.cuh"
#include "Symbol/Symbol.cuh"
#include "Utils/CompileConstants.cuh"
#include "Utils/Order.cuh"

namespace Sym::Heuristic {
    namespace {
        __host__ __device__ void extract_second_factor(const Symbol& product,
                                                       const Symbol& first_factor,
                                                       Symbol& destination, Symbol& help_space) {
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
                    Symbol::copy_and_reverse_symbol_sequence(current_dst, product_it.current(),
                                                             product_it.current()->size());
                    current_dst += current_dst->size();
                    ++expressions_copied;
                    product_it.advance();
                    break;
                case Util::Order::Less:
                    Symbol::copy_and_reverse_symbol_sequence(current_dst, first_factor_it.current(),
                                                             first_factor_it.current()->size());
                    current_dst += current_dst->size();
                    ++expressions_copied;
                    first_factor_it.advance();
                    break;
                }
            }

            auto* const valid_it = product_it.is_valid() ? &product_it : &first_factor_it;
            while (valid_it->is_valid()) {
                Symbol::copy_and_reverse_symbol_sequence(current_dst, valid_it->current(),
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

    __device__ CheckResult is_simple_function(const Integral& integral) {
        if (integral.integrand()->size() == 2) {
            return {1, 1};
        }

        return {0, 0};
    }

    __device__ void integrate_simple_function_by_parts(
        const SubexpressionCandidate& integral, const ExpressionArray<>::Iterator& integral_dst,
        const ExpressionArray<>::Iterator& expression_dst, Symbol& help_space) {
        integrate_by_parts(integral, Static::one(), Static::identity(), integral_dst,
                           expression_dst, help_space);
    }

    __device__ void integrate_by_parts(const SubexpressionCandidate& integral,
                                       const Symbol& first_function_derivative,
                                       const Symbol& first_function,
                                       const ExpressionArray<>::Iterator& integral_dst,
                                       const ExpressionArray<>::Iterator& expression_dst,
                                       Symbol& help_space) {
        const auto& integrand = *integral.arg().as<Integral>().integrand();

        if (integrand.is(Type::Product)) {
            const auto& product = integrand.as<Product>();
            const size_t second_factor_size = product.size - first_function_derivative.size() - 1;

            Candidate<Add<Neg<SingleIntegralVacancy>, Mul<Copy, Skip>>>::init(
                *expression_dst, {{integral.vacancy_expression_idx, integral.vacancy_idx, 1},
                                  first_function,
                                  second_factor_size});

            Symbol& second_factor_dst = (*expression_dst)
                                            .as<SubexpressionCandidate>()
                                            .arg()
                                            .as<Addition>()
                                            .arg2()
                                            .as<Product>()
                                            .arg2();

            extract_second_factor(product, first_function_derivative, second_factor_dst,
                                  help_space);
        }

        if constexpr (Consts::DEBUG) {
            if (second_factor_dst.size() != second_factor_size) {
                Util::crash("Second factor size (%lu) does not match predicted size (%lu)",
                            second_factor_dst.size(), second_factor_size);
            }

            if (second_factor_dst.is(Type::Unknown)) {
                Util::crash("Second factor type is (still) Unknown");
            }
        }

        second_factor_dst.derivative_to(help_space);

        const auto& original_integral = integral.arg().as<Integral>();

        Candidate<Int<Mul<Copy, Copy>>>::init(
            *integral_dst,
            {{expression_dst.index(), 3, 0}, original_integral, first_function, help_space});
    }
}