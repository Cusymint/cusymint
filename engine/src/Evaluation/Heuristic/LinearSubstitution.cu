#include "LinearSubstitution.cuh"

#include "Evaluation/StaticFunctions.cuh"
#include "Symbol/MetaOperators.cuh"
#include "Symbol/Symbol.cuh"
#include "Symbol/TreeIterator.cuh"
#include "Utils/Meta.cuh"
#include "Utils/Pair.cuh"

namespace Sym::Heuristic {
    __device__ CheckResult is_function_of_linear_function(const Integral& integral,
                                                          Symbol& help_space) {
        const auto& integrand = integral.integrand();

        if (integrand.size() == 1) {
            return {0, 0};
        }

        const Symbol* linear_expr = nullptr;

        for (size_t i = 0; i < integrand.size(); ++i) {
            if (AnyOf<Add<Const, Mul<Const, Var>>, Mul<Const, Var>, Add<Const, Var>>::match(
                    *integrand.at(i))) {
                linear_expr = integrand.at(i);
                break;
            }
        }

        if (linear_expr != nullptr) {
            for (ConstTreeIterator<Product> iterator(&integrand); iterator.is_valid();
                 iterator.advance()) {
                if (iterator.current()->is(Type::Variable)) {
                    continue;
                }
                if (iterator.current()->is(Type::Power)) {
                    const auto& power = iterator.current()->as<Power>();
                    if (power.arg1().is(Type::Variable) && power.arg2().is_integer() &&
                        power.arg2().as<NumericConstant>().value > 0) {
                        continue;
                    }
                }
                if (!iterator.current()->is_function_of(&help_space, *linear_expr)) {
                    return {0, 0};
                }
            }
            return {1, 0};
        }

        return {0, 0};
    }

    __device__ EvaluationStatus substitute_linear_function(
        const SubexpressionCandidate& integral, const ExpressionArray<>::Iterator& integral_dst,
        const ExpressionArray<>::Iterator& /*expression_dst*/,
        const ExpressionArray<>::Iterator& help_space) {

        SymbolIterator iterator =
            TRY_EVALUATE_RESULT(SymbolIterator::from_at(*integral_dst, 0, integral_dst.capacity()));

        const auto& integrand = integral.arg().as<Integral>().integrand();

        const Symbol* linear_expr = nullptr;
        const Symbol* linear_coef = nullptr;

        for (size_t i = 0; i < integrand.size(); ++i) {
            if (Add<Const, Mul<Const, Var>>::match(*integrand.at(i))) {
                linear_expr = integrand.at(i);
                linear_coef = &integrand.at(i)->as<Addition>().arg2().as<Product>().arg1();
                Frac<Sub<Var, Copy>, Copy>::init(
                    *help_space, {integrand.at(i)->as<Addition>().arg1(), *linear_coef});
                break;
            }
            if (Mul<Const, Var>::match(*integrand.at(i))) {
                linear_expr = integrand.at(i);
                linear_coef = &integrand.at(i)->as<Product>().arg1();
                Frac<Var, Copy>::init(*help_space, {*linear_coef});
                break;
            }
            if (Add<Const, Var>::match(*integrand.at(i))) {
                linear_expr = integrand.at(i);
                linear_coef = &Static::one();
                Sub<Var, Copy>::init(*help_space, {integrand.at(i)->as<Addition>().arg1()});
                break;
            }
        }

        SubexpressionCandidate* new_candidate = *integral_dst << SubexpressionCandidate::builder();
        new_candidate->copy_metadata_from(integral);
        TRY_EVALUATE_RESULT(iterator += 1);

        const Util::Pair<const Symbol*, const Symbol*> linear_substitutions[] = {
            Util::Pair<const Symbol*, const Symbol*>(std::move(linear_expr), &Static::identity()),
            Util::Pair<const Symbol*, const Symbol*>(&Static::identity(), &*help_space),
        };

        const auto substitution_result =
            integral.arg().as<Integral>().integrate_by_substitution_with_derivative(
                linear_substitutions, 2, *linear_coef, iterator);

        TRY_EVALUATE_RESULT(substitution_result);

        new_candidate->seal();

        return EvaluationStatus::Done;
    }
}