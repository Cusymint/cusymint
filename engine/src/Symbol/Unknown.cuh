#ifndef UNKNOWN_CUH
#define UNKNOWN_CUH

#include "Macros.cuh"

namespace Sym {
    DECLARE_SYMBOL(Unknown, true)
    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::string to_tex() const;
    END_DECLARE_SYMBOL(Unknown)
}

#endif
