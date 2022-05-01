#include "symbol.cuh"

namespace Sym {
    void Symbol::substitute_variable_with(Symbol symbol) {
        for (size_t i = 0; i < unknown.total_size; ++i) {
            if (this[i].is(Type::Variable)) {
                this[i] = symbol;
            }
        }
    }

    std::string Symbol::to_string() { return VIRTUAL_CALL(*this, to_string); }
};
