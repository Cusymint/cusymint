#ifndef SIMPLE_SINE_COSINE_CUH
#define SIMPLE_SINE_COSINE_CUH

#include "KnownIntegral.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_sine(const Integral& integral);
    __device__ EvaluationStatus
    integrate_simple_sine(const Integral& integral, SymbolIterator& destination,
                          const ExpressionArray<>::Iterator& /*help_space*/);

    __device__ size_t is_simple_cosine(const Integral& integral);
    __device__ EvaluationStatus
    integrate_simple_cosine(const Integral& integral, SymbolIterator& destination,
                            const ExpressionArray<>::Iterator& /*help_space*/);
}

#endif
