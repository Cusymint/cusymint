#ifndef SUBSTITUTE_E_TO_X_CUH
#define SUBSTITUTE_E_TO_X_CUH

#include "Heuristic.cuh"

namespace Sym::Heuristic {
    __device__ CheckResult is_function_of_ex(const Integral* const integral);
    __device__ void transform_function_of_ex(const SubexpressionCandidate& integral,
                                             const ExpressionArray<>::Iterator& integral_dst,
                                             const ExpressionArray<>::Iterator& expression_dst,
                                             Symbol& help_space);
}
#endif
