#ifndef SUBSTITUTE_E_TO_X_CUH
#define SUBSTITUTE_E_TO_X_CUH

#include "Heuristic.cuh"

namespace Sym::Heuristic {
    __device__ CheckResult is_function_of_ex(const Integral* const integral);
    __device__ void transform_function_of_ex(const SubexpressionCandidate* const integral,
                                             Symbol* const integral_dst,
                                             Symbol* const /*expression_dst*/,
                                             const size_t /*expression_index*/,
                                             Symbol* const help_space);
}
#endif
