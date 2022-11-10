#ifndef POWER_CUH
#define POWER_CUH

#include <vector>

#include "Macros.cuh"

namespace Sym {
    DECLARE_SYMBOL(Power, false)
    TWO_ARGUMENT_OP_SYMBOL

    std::string to_string() const;
    std::string to_tex() const;

    __host__ __device__ ssize_t is_polynomial(const ssize_t* const ranks) const;
    DEFINE_IS_MONOMIAL(1) // we assume that `this` is a polynomial

    END_DECLARE_SYMBOL(Power)

    std::vector<Symbol> operator^(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs);
    std::vector<Symbol> sqrt(const std::vector<Symbol>& arg);
}

#endif
