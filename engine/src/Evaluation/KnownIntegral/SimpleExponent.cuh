#ifndef SIMPLE_EXPONENT_CUH
#define SIMPLE_EXPONENT_CUH

#include "KnownIntegral.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_exponent(const Integral* const integral);
    __device__ void integrate_simple_exponent(const Integral* const integral,
                                              Symbol* const destination,
                                              Symbol* const /*help_space*/);
}

#endif
