#ifndef HYPERBOLIC_H
#define HYPERBOLIC_H

#include <vector>
#include "Macros.cuh"

namespace Sym {
    DECLARE_SYMBOL(SineHyperbolic, false)
    ONE_ARGUMENT_OP_SYMBOL

    std::string to_string() const;
    END_DECLARE_SYMBOL(SineHyperbolic)

    DECLARE_SYMBOL(CosineHyperbolic, false)
    ONE_ARGUMENT_OP_SYMBOL

    std::string to_string() const;
    END_DECLARE_SYMBOL(CosineHyperbolic)

    DECLARE_SYMBOL(TangentHyperbolic, false)
    ONE_ARGUMENT_OP_SYMBOL

    std::string to_string() const;
    END_DECLARE_SYMBOL(TangentHyperbolic)

    DECLARE_SYMBOL(CotangentHyperbolic, false)
    ONE_ARGUMENT_OP_SYMBOL

    std::string to_string() const;
    END_DECLARE_SYMBOL(CotangentHyperbolic)

    std::vector<Symbol> sinh(const std::vector<Symbol>& arg);
    std::vector<Symbol> cosh(const std::vector<Symbol>& arg);
    std::vector<Symbol> tanh(const std::vector<Symbol>& arg);
    std::vector<Symbol> coth(const std::vector<Symbol>& arg);
}

#endif
