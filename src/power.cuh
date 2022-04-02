#ifndef POWER_CUH
#define POWER_CUH

#include <vector>

#include "symbol_defs.cuh"

namespace Sym {
    DECLARE_SYMBOL(Power, false)
    TWO_ARGUMENT_OP_SYMBOL
    END_DECLARE_SYMBOL(Power)

    std::vector<Symbol> operator^(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs);
} // namespace Sym

#endif
