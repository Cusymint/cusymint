#ifndef INTEGRAL_CUH
#define INTEGRAL_CUH

#include <vector>

#include "symbol_defs.cuh"

namespace Sym {
    DECLARE_SYMBOL(Integral, false)
    size_t substitution_count;
    size_t integrand_offset;
    std::string to_string();
    __host__ __device__ Symbol* integrand();
    END_DECLARE_SYMBOL(Integral)

    std::vector<Symbol> integral(const std::vector<Symbol>& arg);
}

#endif
