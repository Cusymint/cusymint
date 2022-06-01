#include "variable.cuh"

#include "symbol.cuh"

namespace Sym {
    DEFINE_SIMPLE_COMPRESS_REVERSE_TO(Variable)
    DEFINE_SIMPLE_COMPARE(Variable)
    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(Variable);

    std::vector<Symbol> var() {
        std::vector<Symbol> v(1);
        v[0].variable = Variable::create();
        return v;
    }
}
