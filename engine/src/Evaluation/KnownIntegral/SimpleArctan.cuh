#ifndef SIMPLE_ARCTAN_CUH
#define SIMPLE_ARCTAN_CUH

#include "KnownIntegral.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_arctan(const Integral& integral);
    __device__ EvaluationStatus integrate_simple_arctan(
        const Integral& integral, const ExpressionArray<>::Iterator& destination,
        const ExpressionArray<>::Iterator& /*help_space*/);
}

#endif
