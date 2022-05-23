#ifndef INVERSE_TRIGONOMETRIC_CUH
#define INVERSE_TRIGONOMETRIC_CUH

#include <vector>

#include "symbol_defs.cuh"

namespace Sym {
    DECLARE_SYMBOL(Arcsine, false)
    ONE_ARGUMENT_OP_SYMBOL

    std::string to_string() const;
    END_DECLARE_SYMBOL(Arcsine)

    DECLARE_SYMBOL(Arccosine, false)
    ONE_ARGUMENT_OP_SYMBOL

    std::string to_string() const;
    END_DECLARE_SYMBOL(Arccosine)
    
    DECLARE_SYMBOL(Arctangent, false)
    ONE_ARGUMENT_OP_SYMBOL

    std::string to_string() const;
    END_DECLARE_SYMBOL(Arctangent)

    DECLARE_SYMBOL(Arccotangent, false)
    ONE_ARGUMENT_OP_SYMBOL

    std::string to_string() const;
    END_DECLARE_SYMBOL(Arccotangent)

    std::vector<Symbol> arcsin(const std::vector<Symbol>& arg);
    std::vector<Symbol> arccos(const std::vector<Symbol>& arg);
    std::vector<Symbol> arctan(const std::vector<Symbol>& arg);
    std::vector<Symbol> arccot(const std::vector<Symbol>& arg);
}

#endif
