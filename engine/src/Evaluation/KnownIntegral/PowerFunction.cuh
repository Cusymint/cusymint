#ifndef POWER_FUNCTION_CUH
#define POWER_FUNCTION_CUH

#include "KnownIntegral.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_power_function(const Integral* const integral);
    __device__ void integrate_power_function(const Integral* const integral,
                                             Symbol* const destination,
                                             Symbol* const /*help_space*/);
}

#endif
