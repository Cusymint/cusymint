#ifndef TRIGONOMETRIC_CUH
#define TRIGONOMETRIC_CUH

#include <vector>

#include "symbol_defs.cuh"

namespace Sym {
    DECLARE_SYMBOL(Sine, false)
    ONE_ARGUMENT_OP_SYMBOL

    std::string to_string() const;
    END_DECLARE_SYMBOL(Sine)

    DECLARE_SYMBOL(Cosine, false)
    ONE_ARGUMENT_OP_SYMBOL

    std::string to_string() const;
    END_DECLARE_SYMBOL(Cosine)

    DECLARE_SYMBOL(Tangent, false)
    ONE_ARGUMENT_OP_SYMBOL

    std::string to_string() const;
    END_DECLARE_SYMBOL(Tangent)

    DECLARE_SYMBOL(Cotangent, false)
    ONE_ARGUMENT_OP_SYMBOL

    std::string to_string() const;
    END_DECLARE_SYMBOL(Cotangent)

    std::vector<Symbol> sin(const std::vector<Symbol>& arg);
    std::vector<Symbol> cos(const std::vector<Symbol>& arg);
    std::vector<Symbol> tan(const std::vector<Symbol>& arg);
    std::vector<Symbol> cot(const std::vector<Symbol>& arg);
}

#endif
