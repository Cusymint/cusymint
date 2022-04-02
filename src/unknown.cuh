#ifndef UNKNOWN_CUH
#define UNKNOWN_CUH

#include "symbol_defs.cuh"

namespace Sym {
    DECLARE_SYMBOL(Unknown, true)
    std::string to_string() { return "Unknown"; }
    END_DECLARE_SYMBOL(Unknown)
} // namespace Sym

#endif
