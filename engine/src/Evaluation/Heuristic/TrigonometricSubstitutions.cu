#include "TrigonometricSubstitutions.cuh"

#include "Evaluation/StaticFunctions.cuh"
#include "Evaluation/Status.cuh"
#include "Utils/Meta.cuh"

namespace Sym::Heuristic {
    namespace {
        __device__ EvaluationStatus substitute_simple_trig(
            const SubexpressionCandidate& integral, const ExpressionArray<>::Iterator& integral_dst,
            const Util::Pair<const Symbol*, const Symbol*> substitution_pairs[],
            const size_t substitution_count, const Symbol& derivative) {
            SymbolIterator iterator = TRY_EVALUATE_RESULT(
                SymbolIterator::from_at(*integral_dst, 0, integral_dst.capacity()));

            SubexpressionCandidate* new_candidate = *iterator << SubexpressionCandidate::builder();
            new_candidate->copy_metadata_from(integral);

            TRY_EVALUATE_RESULT(iterator += 1);

            const auto substitution_result =
                integral.arg().as<Integral>().integrate_by_substitution_with_derivative(
                    substitution_pairs, substitution_count, derivative, iterator);

            TRY_EVALUATE(result_to_evaluation_status(substitution_result));

            new_candidate->seal();

            return EvaluationStatus::Done;
        }
    }

    __device__ CheckResult is_function_of_simple_trigs(const Integral& integral, Symbol& help_space) {
        const bool is_function_of_simple_trigs = integral.integrand().is_function_of(&help_space, 
            Static::sin_x(), Static::cos_x(), Static::tan_x(), Static::cot_x());
        return {is_function_of_simple_trigs ? 1UL : 0UL, 0UL};
    }

    __device__ EvaluationStatus substitute_sine(
        const SubexpressionCandidate& integral, const ExpressionArray<>::Iterator& integral_dst,
        const ExpressionArray<>::Iterator& /*expression_dst*/,
        const ExpressionArray<>::Iterator& /*help_space*/) {
        const Util::Pair<const Symbol*, const Symbol*> substitution_pairs[] = {
            Util::Pair<const Symbol*, const Symbol*>(&Static::sin_x(), &Static::identity()),
            Util::Pair<const Symbol*, const Symbol*>(&Static::cos_x(),
                                                     &Static::pythagorean_sin_cos()),
            Util::Pair<const Symbol*, const Symbol*>(&Static::tan_x(), &Static::tangent_as_sine()),
            Util::Pair<const Symbol*, const Symbol*>(&Static::cot_x(),
                                                     &Static::cotangent_as_sine()),
        };

        return substitute_simple_trig(integral, integral_dst, substitution_pairs,
                                      Util::array_len(substitution_pairs),
                                      Static::pythagorean_sin_cos());
    }

    __device__ EvaluationStatus substitute_cosine(
        const SubexpressionCandidate& integral, const ExpressionArray<>::Iterator& integral_dst,
        const ExpressionArray<>::Iterator& /*expression_dst*/,
        const ExpressionArray<>::Iterator& /*help_space*/) {
        const Util::Pair<const Symbol*, const Symbol*> substitution_pairs[] = {
            Util::Pair<const Symbol*, const Symbol*>(&Static::cos_x(), &Static::identity()),
            Util::Pair<const Symbol*, const Symbol*>(&Static::sin_x(),
                                                     &Static::pythagorean_sin_cos()),
            Util::Pair<const Symbol*, const Symbol*>(&Static::cot_x(), &Static::tangent_as_sine()),
            Util::Pair<const Symbol*, const Symbol*>(&Static::tan_x(),
                                                     &Static::cotangent_as_sine()),
        };

        return substitute_simple_trig(integral, integral_dst, substitution_pairs,
                                      Util::array_len(substitution_pairs),
                                      Static::neg_pythagorean_sin_cos());
    }

    __device__ EvaluationStatus substitute_tangent(
        const SubexpressionCandidate& integral, const ExpressionArray<>::Iterator& integral_dst,
        const ExpressionArray<>::Iterator& /*expression_dst*/,
        const ExpressionArray<>::Iterator& /*help_space*/) {
        const Util::Pair<const Symbol*, const Symbol*> substitution_pairs[] = {
            Util::Pair<const Symbol*, const Symbol*>(&Static::tan_x(), &Static::identity()),
            Util::Pair<const Symbol*, const Symbol*>(&Static::cot_x(), &Static::inverse()),
            Util::Pair<const Symbol*, const Symbol*>(&Static::sin_x(), &Static::sine_as_tangent()),
            Util::Pair<const Symbol*, const Symbol*>(&Static::cos_x(),
                                                     &Static::cosine_as_tangent()),
        };

        return substitute_simple_trig(integral, integral_dst, substitution_pairs,
                                      Util::array_len(substitution_pairs),
                                      Static::tangent_derivative());
    }
}
