#ifndef ERROR_FUNCTION_CUH
#define ERROR_FUNCTION_CUH

#include <vector>

#include "Macros.cuh"

namespace Sym {
    DECLARE_SYMBOL(ErrorFunction, false)
    ONE_ARGUMENT_OP_SYMBOL

    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::string to_tex() const;
    END_DECLARE_SYMBOL(ErrorFunction)

    std::vector<Symbol> erf(const std::vector<Symbol>& arg);
}

#endif