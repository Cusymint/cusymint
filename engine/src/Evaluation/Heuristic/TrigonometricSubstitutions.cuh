#ifndef TRIGONOMETRIC_SUBSTITUTIONS_CUH
#define TRIGONOMETRIC_SUBSTITUTIONS_CUH

#include "Heuristic.cuh"

namespace Sym::Heuristic {
    __device__ CheckResult is_function_of_simple_trigs(const Integral& integral);
    __device__ EvaluationStatus substitute_sine(const SubexpressionCandidate& integral,
                                    const ExpressionArray<>::Iterator& integral_dst,
                                    const ExpressionArray<>::Iterator& /*expression_dst*/,
                                    const ExpressionArray<>::Iterator& help_space);
    __device__ EvaluationStatus substitute_cosine(const SubexpressionCandidate& integral,
                                      const ExpressionArray<>::Iterator& integral_dst,
                                      const ExpressionArray<>::Iterator& /*expression_dst*/,
                                      const ExpressionArray<>::Iterator& help_space);
    __device__ EvaluationStatus substitute_tangent(const SubexpressionCandidate& integral,
                                       const ExpressionArray<>::Iterator& integral_dst,
                                       const ExpressionArray<>::Iterator& /*expression_dst*/,
                                       const ExpressionArray<>::Iterator& help_space);
}

#endif
