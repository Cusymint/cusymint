#include "Symbol/Macros.cuh"
#include "Variable.cuh"

#include "Symbol.cuh"

namespace Sym {
    DEFINE_SIMPLE_COMPRESS_REVERSE_TO(Variable)
    DEFINE_SIMPLE_COMPARE(Variable)
    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(Variable);
    DEFINE_NO_OP_PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE(Variable)
    DEFINE_SIMPLE_SEAL_WHOLE(Variable)

    DEFINE_IS_FUNCTION_OF(Variable) {
        for (size_t i = 0; i < expression_count; ++i) {
            if (!expressions[i]->is(Type::Variable)) {
                return false;
            }
        }

        return true;
    }

    std::vector<Symbol> var() {
        std::vector<Symbol> var(1);
        var[0].variable = Variable::create();
        return var;
    }
}
