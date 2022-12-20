#include "Symbol/Macros.cuh"
#include "Variable.cuh"

#include "Symbol.cuh"

namespace Sym {
    DEFINE_ZERO_ARGUMENT_OP_FUNCTIONS(Variable)
    DEFINE_SIMPLE_COMPRESS_REVERSE_TO(Variable)
    DEFINE_SIMPLE_ARE_EQUAL(Variable)
    DEFINE_IDENTICAL_COMPARE_TO(Variable)
    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(Variable);
    DEFINE_NO_OP_PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE(Variable)
    DEFINE_NO_OP_PUSH_CHILDREN_ONTO_STACK(Variable)
    DEFINE_SIMPLE_SEAL_WHOLE(Variable)

    DEFINE_IS_FUNCTION_OF(Variable) {
        for (size_t i = 0; i < expression_count; ++i) {
            if (!expressions[i]->is(Type::Variable)) {
                return false;
            }
        }

        return true;
    }

    DEFINE_INSERT_REVERSED_DERIVATIVE_AT(Variable) {
        destination->init_from(NumericConstant::with_value(1));
        return 1;
    }

    std::vector<Symbol> var() {
        std::vector<Symbol> var(1);
        var[0].init_from(Variable::create());
        return var;
    }
}
