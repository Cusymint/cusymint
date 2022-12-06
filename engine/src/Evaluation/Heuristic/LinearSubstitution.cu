#include "LinearSubstitution.cuh"

#include "Evaluation/StaticFunctions.cuh"
#include "Symbol/MetaOperators.cuh"
#include "Symbol/Symbol.cuh"
#include "Utils/Meta.cuh"

namespace Sym::Heuristic {
    __device__ CheckResult is_function_of_linear_function(const Integral& integral) {
        const auto& integrand = *integral.integrand();

        if (integrand.size() == 1) {
            return {0, 0};
        }

        const Symbol* linear_expr = nullptr;

        for (size_t i = 0; i < integrand.size(); ++i) {
            if (Add<Const, Mul<Const, Var>>::match(*integrand.at(i))) {
                linear_expr = integrand.at(i);
                break;
            }
        }

        if (linear_expr != nullptr && integrand.is_function_of(*linear_expr)) {
            return {1, 0};
        }

        return {0, 0};
    }

    __device__ void substitute_linear_function(
        const SubexpressionCandidate& integral, const ExpressionArray<>::Iterator& integral_dst,
        const ExpressionArray<>::Iterator& /*expression_dst*/, Symbol& help_space) {
        const auto& integrand = *integral.arg().as<Integral>().integrand();

        const Symbol* linear_expr = nullptr;
        const Symbol* linear_coef = nullptr;

        for (size_t i = 0; i < integrand.size(); ++i) {
            if (Add<Const, Mul<Const, Var>>::match(*integrand.at(i))) {
                linear_expr = integrand.at(i);
                // free_term = &integrand.at(i)->as<Addition>().arg1();
                linear_coef = &integrand.at(i)->as<Addition>().arg2().as<Product>().arg1();
                break;
            }
        }

        SubexpressionCandidate* new_candidate = *integral_dst << SubexpressionCandidate::builder();
        new_candidate->copy_metadata_from(integral);

        integral.arg().as<Integral>().integrate_by_substitution_with_derivative(
            *linear_expr, *linear_coef, new_candidate->arg());

        new_candidate->seal();
    }
}