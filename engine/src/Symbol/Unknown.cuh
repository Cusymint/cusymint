#ifndef UNKNOWN_CUH
#define UNKNOWN_CUH

#include "Macros.cuh"

namespace Sym {
    DECLARE_SYMBOL(Unknown, true)
    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::string to_tex() const;
    DEFINE_IS_NOT_POLYNOMIAL
    END_DECLARE_SYMBOL(Unknown)
}

#endif
