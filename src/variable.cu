#include "variable.cuh"

#include "symbol.cuh"

namespace Sym {
    std::vector<Symbol> var() {
        std::vector<Symbol> v(1);
        v[0].variable = Variable::create();
        return v;
    }
} // namespace
