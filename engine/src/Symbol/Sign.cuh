#ifndef SIGN_CUH
#define SIGN_CUH

#include <vector>

#include "Macros.cuh"

namespace Sym {
    DECLARE_SYMBOL(Sign, false)
    ONE_ARGUMENT_OP_SYMBOL

    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::string to_tex() const;
    END_DECLARE_SYMBOL(Sign)

    [[nodiscard]] std::vector<Symbol> sgn(const std::vector<Symbol>& arg);
    [[nodiscard]] std::vector<Symbol> abs(const std::vector<Symbol>& arg);
}

#endif
