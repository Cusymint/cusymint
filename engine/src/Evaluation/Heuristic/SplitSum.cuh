#ifndef SPLIT_SUM_CUH
#define SPLIT_SUM_CUH

#include "Heuristic.cuh"

namespace Sym::Heuristic {
    __device__ CheckResult is_sum(const Integral& integral, Symbol& help_space);
    __device__ void split_sum(const SubexpressionCandidate& integral,
                              const ExpressionArray<>::Iterator& integral_dst,
                              const ExpressionArray<>::Iterator& expression_dst,
                              Symbol& /*help_space*/);
}

#endif
