#ifndef LINEAR_SUBSTITUTION_CUH
#define LINEAR_SUBSTITUTION_CUH

#include "Heuristic.cuh"

namespace Sym::Heuristic {
    __device__ CheckResult is_function_of_linear_function(const Integral& integral, Symbol& help_space);
    __device__ EvaluationStatus substitute_linear_function(const SubexpressionCandidate& integral,
                                          const ExpressionArray<>::Iterator& integral_dst,
                                          const ExpressionArray<>::Iterator& expression_dst,
                                          const ExpressionArray<>::Iterator& help_space);
}
#endif