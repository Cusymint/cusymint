#ifndef SPLIT_SUM_CUH
#define SPLIT_SUM_CUH

#include "Heuristic.cuh"

namespace Sym::Heuristic {
    __device__ CheckResult is_sum(const Integral* const integral);
    __device__ void split_sum(const SubexpressionCandidate* const integral,
                              Symbol* const integral_dst, Symbol* const expression_dst,
                              const size_t expression_index, Symbol* const /*help_space*/);
}

#endif
