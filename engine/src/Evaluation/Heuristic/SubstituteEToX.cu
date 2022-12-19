#include "SubstituteEToX.cuh"

#include "Evaluation/StaticFunctions.cuh"

namespace Sym::Heuristic {
    __device__ CheckResult is_function_of_ex(const Integral& integral) {
        return {integral.integrand().is_function_of(Static::e_to_x()) ? 1UL : 0UL, 0UL};
    }

    __device__ EvaluationStatus transform_function_of_ex(
        const SubexpressionCandidate& integral, const ExpressionArray<>::Iterator& integral_dst,
        const ExpressionArray<>::Iterator& /*expression_dst*/,
        const ExpressionArray<>::Iterator& /*help_space*/) {
        SymbolIterator iterator =
            TRY_EVALUATE_RESULT(SymbolIterator::from_at(*integral_dst, 0, integral_dst.capacity()));

        auto* new_candidate = *iterator << SubexpressionCandidate::builder();
        new_candidate->copy_metadata_from(integral);
        TRY_EVALUATE_RESULT(iterator += 1);

        const auto substitution_result =
            integral.arg().as<Integral>().integrate_by_substitution_with_derivative(
                Static::e_to_x(), Static::identity(), iterator);

        TRY_EVALUATE_RESULT(substitution_result);

        new_candidate->seal();

        return EvaluationStatus::Done;
    }
}
