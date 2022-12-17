#ifndef X_INVERSE_CUH
#define X_INVERSE_CUH

#include "KnownIntegral.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_x_inverse(const Integral& integral);
    __device__ EvaluationStatus
    integrate_x_inverse(const Integral& integral, SymbolIterator& destination,
                        const ExpressionArray<>::Iterator& /*help_space*/);
}

#endif
