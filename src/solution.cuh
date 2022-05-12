#ifndef SOLUTION_CUH
#define SOLUTION_CUH

#include <vector>

#include "symbol_defs.cuh"

namespace Sym {
    DECLARE_SYMBOL(Solution, false)
    size_t substitution_count;
    size_t symbols_offset;
    std::string to_string() const;
    END_DECLARE_SYMBOL(Solution)

    std::vector<Symbol> solution(const std::vector<Symbol>& arg);
}

#endif
