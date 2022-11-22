#ifndef CONSTANT_INTEGRAL_CUH
#define CONSTANT_INTEGRAL_CUH

#include "KnownIntegral.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_constant_integral(const Integral& integral);
    __device__ EvaluationStatus integrate_constant_integral(const Integral& integral, Symbol& destination,
                                                Symbol& /*help_space*/);
}

#endif
