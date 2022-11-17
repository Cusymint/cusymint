#ifndef LOGARITHM_CUH
#define LOGARITHM_CUH

#include <vector>

#include "Macros.cuh"

namespace Sym {
    DECLARE_SYMBOL(Logarithm, false)
    ONE_ARGUMENT_OP_SYMBOL

    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::string to_tex() const;
    END_DECLARE_SYMBOL(Logarithm)

    std::vector<Symbol> log(const std::vector<Symbol>& base, const std::vector<Symbol>& arg);
    std::vector<Symbol> ln(const std::vector<Symbol>& arg);
}

#endif