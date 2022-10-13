#ifndef POWER_CUH
#define POWER_CUH

#include <vector>

#include "Macros.cuh"

namespace Sym {
    DECLARE_SYMBOL(Power, false)
    TWO_ARGUMENT_OP_SYMBOL

    std::string to_string() const;
    std::string to_tex() const;

    __host__ __device__ int is_polynomial() const;
    __host__ __device__ double get_monomial_coefficient() const;

    END_DECLARE_SYMBOL(Power)

    std::vector<Symbol> operator^(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs);
}

#endif
