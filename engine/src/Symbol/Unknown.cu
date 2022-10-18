#include "Symbol/Unknown.cuh"

#include "Symbol/Symbol.cuh"
#include <fmt/core.h>

namespace Sym {
    DEFINE_SIMPLE_COMPARE(Unknown);
    DEFINE_SIMPLE_COMPRESS_REVERSE_TO(Unknown);
    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(Unknown);

    [[nodiscard]] std::string Unknown::to_string() const {
        return fmt::format("Unknown(type={},size={})",
                           static_cast<std::underlying_type<Type>::type>(type), size);
    }

    [[nodiscard]] std::string Unknown::to_tex() const {
        return fmt::format(R"(?_{{ \text{{ type= }} {}, \text{{ size= }} {} }})",
                           static_cast<std::underlying_type<Type>::type>(type), size);
    }
}
