#ifndef VARIABLE_CUH
#define VARIABLE_CUH

#include <vector>

#include "Macros.cuh"

namespace Sym {
    DECLARE_SYMBOL(Variable, true)
    DEFINE_TO_STRING("x");
    DEFINE_TO_TEX("x");
    END_DECLARE_SYMBOL(Variable)

    std::vector<Symbol> var();
}

#endif
