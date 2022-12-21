#ifndef HEURISTIC_CUH
#define HEURISTIC_CUH

#include "Evaluation/Status.cuh"
#include "Symbol/ExpressionArray.cuh"
#include "Symbol/Symbol.cuh"

namespace Sym::Heuristic {
    struct CheckResult {
        __host__ __device__ CheckResult(const size_t new_integrals, const size_t new_expressions) :
            new_integrals(new_integrals), new_expressions(new_expressions) {}

        __host__ __device__ static inline CheckResult empty() { return {0UL, 0UL}; }

        size_t new_integrals;
        size_t new_expressions;
    };

    using Check = CheckResult (*)(const Integral& integral, Symbol& help_space);
    using Application = EvaluationStatus (*)(const SubexpressionCandidate& integral,
                                             const ExpressionArray<>::Iterator& integral_dst,
                                             const ExpressionArray<>::Iterator& expression_dst,
                                             const ExpressionArray<>::Iterator& help_space);

    extern __device__ const Check CHECKS[];
    extern __device__ const Application APPLICATIONS[];

#ifdef __CUDA_ARCH__
    __device__
#endif
        extern const size_t COUNT;
}

#endif
