#include "Symbol/Macros.cuh"
#include "Symbol/Unknown.cuh"

#include "Symbol/Symbol.cuh"
#include <fmt/core.h>

namespace Sym {
    DEFINE_SIMPLE_ARE_EQUAL(Unknown);
    DEFINE_IDENTICAL_COMPARE_TO(Unknown)
    DEFINE_SIMPLE_COMPRESS_REVERSE_TO(Unknown);
    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(Unknown);
    DEFINE_INVALID_IS_FUNCTION_OF(Unknown);
    DEFINE_NO_OP_PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE(Unknown)
    DEFINE_NO_OP_PUSH_CHILDREN_ONTO_STACK(Unknown)
    DEFINE_INVALID_SEAL_WHOLE(Unknown)

    [[nodiscard]] std::string Unknown::to_string() const {
        return fmt::format("Unknown(type={},size={})",
                           static_cast<std::underlying_type<Type>::type>(type), size);
    }

    [[nodiscard]] std::string Unknown::to_tex() const {
        return fmt::format(R"(?_{{ \text{{ type= }} {}, \text{{ size= }} {} }})",
                           static_cast<std::underlying_type<Type>::type>(type), size);
    }
}
