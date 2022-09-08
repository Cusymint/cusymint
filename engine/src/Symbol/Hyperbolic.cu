#include "Hyperbolic.cuh"
#include "Symbol.cuh"

namespace Sym {
    std::vector<Symbol> sinh(const std::vector<Symbol>& arg) {
        return Sym::num(0.5)*(Sym::e()^arg - Sym::e()^(-arg));
    }

    std::vector<Symbol> cosh(const std::vector<Symbol>& arg) {
        return Sym::num(0.5)*(Sym::e()^arg + Sym::e()^(-arg));
    }

    std::vector<Symbol> tanh(const std::vector<Symbol>& arg) {
        return (Sym::e()^arg - Sym::e()^(-arg))/(Sym::e()^arg + Sym::e()^(-arg));
    }

    std::vector<Symbol> coth(const std::vector<Symbol>& arg) {
        return (Sym::e()^arg + Sym::e()^(-arg))/(Sym::e()^arg - Sym::e()^(-arg));
    }
}