#include "unknown.cuh"

#include "symbol.cuh"

namespace Sym {
    DEFINE_SIMPLE_COMPARE(Unknown);
    DEFINE_SIMPLE_COMPRESS_REVERSE_TO(Unknown);
    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(Unknown);

    std::string Unknown::to_string() const {
        return "Unknown(type=" +
               std::to_string(static_cast<std::underlying_type<Type>::type>(type)) +
               ",size=" + std::to_string(size) + ")";
    }
}
