#ifndef SIMPLE_ARCTAN_ARCSINE_CUH
#define SIMPLE_ARCTAN_ARCSINE_CUH

#include "KnownIntegral.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_arctan(const Integral& integral);
    __device__ EvaluationStatus integrate_simple_arctan(const Integral& integral, SymbolIterator& destination,
                             const ExpressionArray<>::Iterator& /*help_space*/);
    __device__ size_t is_simple_arcsine(const Integral& integral);
    __device__ EvaluationStatus integrate_simple_arcsine(const Integral& integral, SymbolIterator& destination,
                             const ExpressionArray<>::Iterator& /*help_space*/);
}

#endif
