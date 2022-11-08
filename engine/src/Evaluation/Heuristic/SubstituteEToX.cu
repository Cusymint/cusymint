#include "SubstituteEToX.cuh"

#include "Evaluation/StaticFunctions.cuh"

namespace Sym::Heuristic {
    __device__ CheckResult is_function_of_ex(const Integral& integral) {
        return {integral.integrand()->is_function_of(Static::e_to_x()) ? 1UL : 0UL, 0UL};
    }

    __device__ void transform_function_of_ex(const SubexpressionCandidate& integral,
                                             const ExpressionArray<>::Iterator& integral_dst,
                                             const ExpressionArray<>::Iterator& /*expression_dst*/,
                                             Symbol& help_space) {
        auto* new_candidate = *integral_dst << SubexpressionCandidate::builder();
        new_candidate->copy_metadata_from(integral);

        integral.arg().as<Integral>().integrate_by_substitution_with_derivative(
            Static::e_to_x(), Static::identity(), *integral_dst->child(), help_space);

        new_candidate->seal();
    }
}
