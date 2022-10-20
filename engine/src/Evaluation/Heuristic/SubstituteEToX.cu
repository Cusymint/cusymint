#include "SubstituteEToX.cuh"

#include "Evaluation/StaticFunctions.cuh"

namespace Sym::Heuristic {
    __device__ CheckResult is_function_of_ex(const Integral* const integral) {
        return {integral->integrand()->is_function_of(Static::e_to_x()) ? 1UL : 0UL, 0UL};
    }

    __device__ void transform_function_of_ex(const SubexpressionCandidate* const integral,
                                             Symbol* const integral_dst,
                                             Symbol* const /*expression_dst*/,
                                             const size_t /*expression_index*/,
                                             Symbol* const help_space) {
        Symbol variable{};
        variable.variable = Variable::create();

        SubexpressionCandidate* new_candidate = integral_dst << SubexpressionCandidate::builder();
        new_candidate->copy_metadata_from(*integral);

        integral->arg().as<Integral>().integrate_by_substitution_with_derivative(
            Static::e_to_x(), &variable, integral_dst + 1, help_space);

        new_candidate->seal();
    }
}
