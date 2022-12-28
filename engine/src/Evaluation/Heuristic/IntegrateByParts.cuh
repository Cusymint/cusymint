#ifndef INTEGRATE_BY_PARTS_CUH
#define INTEGRATE_BY_PARTS_CUH

#include "Heuristic.cuh"

namespace Sym::Heuristic {
    __device__ CheckResult is_simple_function(const Integral& integral, Symbol& help_space);
    __device__ EvaluationStatus integrate_simple_function_by_parts(
        const SubexpressionCandidate& integral, const ExpressionArray<>::Iterator& integral_dst,
        const ExpressionArray<>::Iterator& expression_dst,
        const ExpressionArray<>::Iterator& help_space);

    __device__ EvaluationStatus integrate_by_parts(
        const SubexpressionCandidate& integral, const Symbol& first_function_derivative,
        const Symbol& first_function, const ExpressionArray<>::Iterator& integral_dst,
        const ExpressionArray<>::Iterator& expression_dst, const ExpressionArray<>::Iterator& help_space);
}

#endif