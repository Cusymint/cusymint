#ifndef POWER_FUNCTION_CUH
#define POWER_FUNCTION_CUH

#include "KnownIntegral.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_power_function(const Integral& integral);
    __device__ EvaluationStatus integrate_power_function(const Integral& integral,
                                             const ExpressionArray<>::Iterator& destination,
                                             const ExpressionArray<>::Iterator& /*help_space*/);
}

#endif
