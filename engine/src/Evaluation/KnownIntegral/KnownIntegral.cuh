#ifndef KNOWN_INTEGRAL_CUH
#define KNOWN_INTEGRAL_CUH

#include "Symbol/Symbol.cuh"

namespace Sym::KnownIntegral {
    using Check = size_t (*)(const Integral* const integral);
    using Application = void (*)(const Integral* const integral, Symbol* const destination,
                                 Symbol* const help_space);

    extern __device__ const Check CHECKS[];
    extern __device__ const Application APPLICATIONS[];

#ifdef __CUDA_ARCH__
    __device__
#endif
        extern const size_t COUNT;

    /*
     * @brief Creates `Solution` and writes it to `destination` together with substitutions from
     * `integral`
     *
     * @param integral Integral from which substitutions are to be copied
     * @param destination Result destination
     *
     * @return Pointer to the symbol behind the last substitution in the result
     */
    __device__ Symbol* prepare_solution(const Integral* const integral, Symbol* const destination);
}

#endif
