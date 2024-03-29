#ifndef SUBSTITUTE_E_TO_X_CUH
#define SUBSTITUTE_E_TO_X_CUH

#include "Heuristic.cuh"

namespace Sym::Heuristic {
    __device__ CheckResult is_function_of_ex(const Integral& integral, Symbol& help_space);
    __device__ EvaluationStatus transform_function_of_ex(
        const SubexpressionCandidate& integral, const ExpressionArray<>::Iterator& integral_dst,
        const ExpressionArray<>::Iterator& expression_dst,
        const ExpressionArray<>::Iterator& help_space);
}
#endif
