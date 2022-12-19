#ifndef SIMPLE_EXPONENT_CUH
#define SIMPLE_EXPONENT_CUH

#include "KnownIntegral.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_exponent(const Integral& integral);
    __device__ EvaluationStatus
    integrate_simple_exponent(const Integral& integral, SymbolIterator& destination,
                              const ExpressionArray<>::Iterator& /*help_space*/);

    __device__ size_t is_power_with_constant_base(const Integral& integral);
    __device__ EvaluationStatus
    integrate_power_with_constant_base(const Integral& integral, SymbolIterator& destination,
                                       const ExpressionArray<>::Iterator& /*help_space*/);
}

#endif
