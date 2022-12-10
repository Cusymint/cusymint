#ifndef NON_ELEMENTARY_CUH
#define NON_ELEMENTARY_CUH

#include "KnownIntegral.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_erf(const Integral& integral);
    __device__ void integrate_simple_erf(const Integral& integral, Symbol& destination,
                                         Symbol& /*help_space*/);

    __device__ size_t is_simple_Si(const Integral& integral);
    __device__ void integrate_simple_Si(const Integral& integral, Symbol& destination,
                                        Symbol& /*help_space*/);

    __device__ size_t is_simple_Ci(const Integral& integral);
    __device__ void integrate_simple_Ci(const Integral& integral, Symbol& destination,
                                        Symbol& /*help_space*/);

    __device__ size_t is_simple_Ei(const Integral& integral);
    __device__ void integrate_simple_Ei(const Integral& integral, Symbol& destination,
                                        Symbol& /*help_space*/);

    __device__ size_t is_simple_li(const Integral& integral);
    __device__ void integrate_simple_li(const Integral& integral, Symbol& destination,
                                        Symbol& /*help_space*/);
}

#endif