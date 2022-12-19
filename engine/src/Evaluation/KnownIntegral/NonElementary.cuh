#ifndef NON_ELEMENTARY_CUH
#define NON_ELEMENTARY_CUH

#include "KnownIntegral.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_erf(const Integral& integral);
    __device__ EvaluationStatus integrate_simple_erf(const Integral& integral, SymbolIterator& destination,
                             const ExpressionArray<>::Iterator& /*help_space*/);

    __device__ size_t is_simple_Si(const Integral& integral);
    __device__ EvaluationStatus integrate_simple_Si(const Integral& integral, SymbolIterator& destination,
                             const ExpressionArray<>::Iterator& /*help_space*/);

    __device__ size_t is_simple_Ci(const Integral& integral);
    __device__ EvaluationStatus integrate_simple_Ci(const Integral& integral, SymbolIterator& destination,
                             const ExpressionArray<>::Iterator& /*help_space*/);

    __device__ size_t is_simple_Ei(const Integral& integral);
    __device__ EvaluationStatus integrate_simple_Ei(const Integral& integral, SymbolIterator& destination,
                             const ExpressionArray<>::Iterator& /*help_space*/);

    __device__ size_t is_simple_li(const Integral& integral);
    __device__ EvaluationStatus integrate_simple_li(const Integral& integral, SymbolIterator& destination,
                             const ExpressionArray<>::Iterator& /*help_space*/);
}

#endif