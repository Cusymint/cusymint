#ifndef VARIABLE_CUH
#define VARIABLE_CUH

#include <vector>

#include "Macros.cuh"

namespace Sym {
    DECLARE_SYMBOL(Variable, true)
    DEFINE_TO_STRING("x");
    DEFINE_TO_TEX("x");
    DEFINE_IS_POLYNOMIAL(1)
    DEFINE_IS_MONOMIAL(1)
    END_DECLARE_SYMBOL(Variable)

    std::vector<Symbol> var();
}

#endif
