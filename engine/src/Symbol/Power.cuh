#ifndef POWER_CUH
#define POWER_CUH

#include <vector>

#include "Macros.cuh"

namespace Sym {
    DECLARE_SYMBOL(Power, false)
    TWO_ARGUMENT_OP_SYMBOL

    __host__ __device__ void simplify_sign_power();

    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::string to_tex() const;

    END_DECLARE_SYMBOL(Power)

    std::vector<Symbol> operator^(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs);
    std::vector<Symbol> sqrt(const std::vector<Symbol>& arg);
}

#endif
