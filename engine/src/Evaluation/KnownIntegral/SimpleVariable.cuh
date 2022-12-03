#ifndef SIMPLE_VARIABLE_CUH
#define SIMPLE_VARIABLE_CUH

#include "KnownIntegral.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_variable(const Integral& integral);
    __device__ EvaluationStatus integrate_simple_variable(
        const Integral& integral, const ExpressionArray<>::Iterator& destination,
        const ExpressionArray<>::Iterator& /*help_space*/);
}

#endif
