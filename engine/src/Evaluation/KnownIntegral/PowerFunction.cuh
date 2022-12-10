#ifndef POWER_FUNCTION_CUH
#define POWER_FUNCTION_CUH

#include "KnownIntegral.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_power_function(const Integral& integral);
    __device__ void integrate_power_function(const Integral& integral, Symbol& destination,
                                             Symbol& /*help_space*/);

    __device__ size_t is_reciprocal(const Integral& integral);
    __device__ void integrate_reciprocal(const Integral& integral, Symbol& destination,
                                         Symbol& /*help_space*/);
}

#endif
