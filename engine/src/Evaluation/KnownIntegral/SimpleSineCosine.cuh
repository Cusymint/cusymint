#ifndef SIMPLE_SINE_COSINE_CUH
#define SIMPLE_SINE_COSINE_CUH

#include "KnownIntegral.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_sine(const Integral* const integral);
    __device__ void integrate_simple_sine(const Integral* const integral, Symbol* const destination,
                                          Symbol* const /*help_space*/);

    __device__ size_t is_simple_cosine(const Integral* const integral);
    __device__ void integrate_simple_cosine(const Integral* const integral,
                                            Symbol* const destination,
                                            Symbol* const /*help_space*/);
}

#endif
