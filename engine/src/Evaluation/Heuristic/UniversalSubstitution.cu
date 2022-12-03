#include "UniversalSubstitution.cuh"

#include "Evaluation/StaticFunctions.cuh"
#include "Utils/Meta.cuh"

namespace Sym::Heuristic {
    __device__ CheckResult is_function_of_trigs(const Integral& integral) {
        const bool is_function_of_trigs =
            integral.integrand().is_function_of(Static::tan_x_over_2(), Static::sin_x(),
                                                Static::cos_x(), Static::tan_x(), Static::cot_x());
        return {is_function_of_trigs ? 1UL : 0UL, 0UL};
    }

    __device__ EvaluationStatus do_universal_substitution(
        const SubexpressionCandidate& integral, const ExpressionArray<>::Iterator& integral_dst,
        const ExpressionArray<>::Iterator& /*expression_dst*/, Symbol& /*help_space*/) {
        const Util::Pair<const Symbol*, const Symbol*> substitution_pairs[] = {
            Util::Pair<const Symbol*, const Symbol*>(&Static::tan_x_over_2(), &Static::identity()),
            Util::Pair<const Symbol*, const Symbol*>(&Static::sin_x(), &Static::universal_sin_x()),
            Util::Pair<const Symbol*, const Symbol*>(&Static::cos_x(), &Static::universal_cos_x()),
            Util::Pair<const Symbol*, const Symbol*>(&Static::tan_x(), &Static::universal_tan_x()),
            Util::Pair<const Symbol*, const Symbol*>(&Static::cot_x(), &Static::universal_cot_x()),
        };

        ENSURE_ENOUGH_SPACE(1, integral_dst)
        SubexpressionCandidate* new_candidate = *integral_dst << SubexpressionCandidate::builder();
        new_candidate->copy_metadata_from(integral);

        SymbolIterator iterator = SymbolIterator::from_at(new_candidate->arg(), )

        const auto substitution_result =
            integral.arg().as<Integral>().integrate_by_substitution_with_derivative(
                substitution_pairs, Util::array_len(substitution_pairs),
                Static::universal_derivative(), new_candidate->arg());
        TRY_EVALUATE(result_to_evaluation_status(substitution_result));

        new_candidate->seal();

        return EvaluationStatus::Done;
    }
}
