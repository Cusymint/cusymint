#ifndef CONSTANT_INTEGRAL_CUH
#define CONSTANT_INTEGRAL_CUH

#include "KnownIntegral.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_constant_integral(const Integral* const integral);
    __device__ void integrate_constant_integral(const Integral* const integral,
                                                Symbol* const destination,
                                                Symbol* const /*help_space*/);
}

#endif
