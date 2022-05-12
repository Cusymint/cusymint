#ifndef PRODUCT_CUH
#define PRODUCT_CUH

#include <vector>

#include "symbol_defs.cuh"

namespace Sym {
    DECLARE_SYMBOL(Product, false)
    TWO_ARGUMENT_OP_SYMBOL

    std::string to_string() const;
    END_DECLARE_SYMBOL(Product)

    DECLARE_SYMBOL(Reciprocal, false)
    ONE_ARGUMENT_OP_SYMBOL

    std::string to_string() const;
    END_DECLARE_SYMBOL(Reciprocal)

    std::vector<Symbol> operator*(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs);
    std::vector<Symbol> operator/(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs);
}

#endif
