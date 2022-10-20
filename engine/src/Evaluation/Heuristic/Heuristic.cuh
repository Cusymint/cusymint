#ifndef HEURISTIC_CUH
#define HEURISTIC_CUH

#include "Symbol/Symbol.cuh"

namespace Sym::Heuristic {
    struct CheckResult {
        __host__ __device__ CheckResult(const size_t new_integrals, const size_t new_expressions) :
            new_integrals(new_integrals), new_expressions(new_expressions) {}

        size_t new_integrals;
        size_t new_expressions;
    };

    using Check = CheckResult (*)(const Integral* const integral);

    using Application = void (*)(const SubexpressionCandidate* const integral,
                                 Symbol* const integral_dst, Symbol* const expression_dst,
                                 const size_t expression_index, Symbol* const help_space);

    extern __device__ const Check CHECKS[];
    extern __device__ const Application APPLICATIONS[];

#ifdef __CUDA_ARCH__
    __device__
#endif
        extern const size_t COUNT;
}

#endif
