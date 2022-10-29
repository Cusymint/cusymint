#ifndef KNOWN_INTEGRAL_CUH
#define KNOWN_INTEGRAL_CUH

#include "Symbol/Symbol.cuh"

namespace Sym::KnownIntegral {
    using Check = size_t (*)(const Integral& integral);
    using Application = void (*)(const Integral& integral, Symbol& destination, Symbol& help_space);

    extern __device__ const Check CHECKS[];
    extern __device__ const Application APPLICATIONS[];

#ifdef __CUDA_ARCH__
    __device__
#endif
        extern const size_t COUNT;
}

#endif
