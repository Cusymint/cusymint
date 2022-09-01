#ifndef UNKNOWN_CUH
#define UNKNOWN_CUH

#include "Macros.cuh"

namespace Sym {
    DECLARE_SYMBOL(Unknown, true)
    std::string to_string() const;
    END_DECLARE_SYMBOL(Unknown)
}

#endif
