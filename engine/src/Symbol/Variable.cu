#include "Symbol/Macros.cuh"
#include "Variable.cuh"

#include "Symbol.cuh"

namespace Sym {
    DEFINE_SIMPLE_COMPRESS_REVERSE_TO(Variable)
    DEFINE_SIMPLE_ARE_EQUAL(Variable)
    DEFINE_IDENTICAL_COMPARE_TO(Variable)
    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(Variable);

    DEFINE_IS_FUNCTION_OF(Variable) {
        for (size_t i = 0; i < expression_count; ++i) {
            if (!expressions[i]->is(Type::Variable)) {
                return false;
            }
        }

        return true;
    }

    DEFINE_NO_OP_PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE(Variable)

    std::vector<Symbol> var() {
        std::vector<Symbol> var(1);
        var[0].variable = Variable::create();
        return var;
    }
}
