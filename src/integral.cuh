#ifndef INTEGRAL_CUH
#define INTEGRAL_CUH

#include <vector>

#include "symbol_defs.cuh"

namespace Sym {
    DECLARE_SYMBOL(Integral, false)
    size_t substitution_count;
    size_t symbols_offset;
    std::string to_string();
    END_DECLARE_SYMBOL(Integral)

    std::vector<Symbol> integral(const std::vector<Symbol>& arg);
}

#endif
