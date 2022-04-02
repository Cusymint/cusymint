#ifndef ADDITION_CUH
#define ADDITION_CUH

#include <vector>

#include "symbol_defs.cuh"

namespace Sym {
    DECLARE_SYMBOL(Addition, false)
    TWO_ARGUMENT_OP_SYMBOL

    std::string to_string();
    END_DECLARE_SYMBOL(Addition)

    DECLARE_SYMBOL(Negative, false)
    ONE_ARGUMENT_OP_SYMBOL

    std::string to_string();
    END_DECLARE_SYMBOL(Negative)

    std::vector<Symbol> operator+(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs);
    std::vector<Symbol> operator-(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs);
    std::vector<Symbol> operator-(const std::vector<Symbol>& arg);
} // namespace Sym

#endif
