#ifndef SIMPLE_TANGENT_COTANGENT_CUH
#define SIMPLE_TANGENT_COTANGENT_CUH

#include "KnownIntegral.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_tangent(const Integral& integral);
    __device__ EvaluationStatus integrate_simple_tangent(const Integral& integral, SymbolIterator& destination,
                             const ExpressionArray<>::Iterator& /*help_space*/);

    __device__ size_t is_simple_cotangent(const Integral& integral);
    __device__ EvaluationStatus integrate_simple_cotangent(const Integral& integral, SymbolIterator& destination,
                             const ExpressionArray<>::Iterator& /*help_space*/);
}

#endif