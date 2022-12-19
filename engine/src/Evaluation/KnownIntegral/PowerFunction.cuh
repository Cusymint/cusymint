#ifndef POWER_FUNCTION_CUH
#define POWER_FUNCTION_CUH

#include "KnownIntegral.cuh"
#include "Symbol/ExpressionArray.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_power_function(const Integral& integral);
    __device__ EvaluationStatus integrate_power_function(const Integral& integral, SymbolIterator& destination,
                             const ExpressionArray<>::Iterator& /*help_space*/);

    __device__ size_t is_reciprocal(const Integral& integral);
    __device__ EvaluationStatus integrate_reciprocal(const Integral& integral, SymbolIterator& destination,
                             const ExpressionArray<>::Iterator& /*help_space*/);
}

#endif
