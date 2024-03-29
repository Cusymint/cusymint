#ifndef HYPERBOLIC_H
#define HYPERBOLIC_H

#include "Macros.cuh"
#include <vector>

namespace Sym {
    std::vector<Symbol> sinh(const std::vector<Symbol>& arg);
    std::vector<Symbol> cosh(const std::vector<Symbol>& arg);
    std::vector<Symbol> tanh(const std::vector<Symbol>& arg);
    std::vector<Symbol> coth(const std::vector<Symbol>& arg);
}

#endif
