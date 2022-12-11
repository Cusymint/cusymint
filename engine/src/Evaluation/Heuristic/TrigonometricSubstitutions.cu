#include "TrigonometricSubstitutions.cuh"

#include "Evaluation/StaticFunctions.cuh"
#include "Utils/Meta.cuh"

namespace Sym::Heuristic {
    __device__ CheckResult is_function_of_simple_trigs(const Integral& integral) {
        const bool is_function_of_simple_trigs = integral.integrand()->is_function_of(
            Static::sin_x(), Static::cos_x(), Static::tan_x(), Static::cot_x());
        return {is_function_of_simple_trigs ? 1UL : 0UL, 0UL};
    }

    __device__ void substitute_sine(const SubexpressionCandidate& integral,
                                    const ExpressionArray<>::Iterator& integral_dst,
                                    const ExpressionArray<>::Iterator& /*expression_dst*/,
                                    Symbol& help_space) {
        const Util::Pair<const Symbol*, const Symbol*> substitution_pairs[] = {
            Util::Pair(&Static::sin_x(), &Static::identity()),
            Util::Pair(&Static::cos_x(), &Static::pythagorean_sin_cos()),
            Util::Pair(&Static::tan_x(), &Static::tangent_as_sine()),
            Util::Pair(&Static::cot_x(), &Static::cotangent_as_sine()),
        };

        SubexpressionCandidate* new_candidate = *integral_dst << SubexpressionCandidate::builder();
        new_candidate->copy_metadata_from(integral);

        integral.arg().as<Integral>().integrate_by_substitution_with_derivative(
            substitution_pairs, Util::array_len(substitution_pairs), Static::pythagorean_sin_cos(),
            new_candidate->arg());

        new_candidate->seal();
    }

    __device__ void substitute_cosine(const SubexpressionCandidate& integral,
                                    const ExpressionArray<>::Iterator& integral_dst,
                                    const ExpressionArray<>::Iterator& /*expression_dst*/,
                                    Symbol& help_space) {
        const Util::Pair<const Symbol*, const Symbol*> substitution_pairs[] = {
            Util::Pair(&Static::cos_x(), &Static::identity()),
            Util::Pair(&Static::sin_x(), &Static::pythagorean_sin_cos()),
            Util::Pair(&Static::cot_x(), &Static::tangent_as_sine()),
            Util::Pair(&Static::tan_x(), &Static::cotangent_as_sine()),
        };

        SubexpressionCandidate* new_candidate = *integral_dst << SubexpressionCandidate::builder();
        new_candidate->copy_metadata_from(integral);

        integral.arg().as<Integral>().integrate_by_substitution_with_derivative(
            substitution_pairs, Util::array_len(substitution_pairs), Static::neg_pythagorean_sin_cos(),
            new_candidate->arg());

        new_candidate->seal();
    }

    // __device__ void substitute_tangent(const SubexpressionCandidate& integral,
    //                                 const ExpressionArray<>::Iterator& integral_dst,
    //                                 const ExpressionArray<>::Iterator& /*expression_dst*/,
    //                                 Symbol& help_space) {
    //     const Util::Pair<const Symbol*, const Symbol*> substitution_pairs[] = {
    //         Util::Pair(&Static::tan_x(), &Static::identity()),
    //         Util::Pair(&Static::cot_x(), &Static::inverse()),
    //         Util::Pair(&Static::sin_x(), &Static::tangent_as_sine()),
    //         Util::Pair(&Static::cos_x(), &Static::cotangent_as_sine()),
    //     };

    //     SubexpressionCandidate* new_candidate = *integral_dst << SubexpressionCandidate::builder();
    //     new_candidate->copy_metadata_from(integral);

    //     integral.arg().as<Integral>().integrate_by_substitution_with_derivative(
    //         substitution_pairs, Util::array_len(substitution_pairs), Static::pythagorean_sin_cos(),
    //         new_candidate->arg());

    //     new_candidate->seal();
    // }
}
