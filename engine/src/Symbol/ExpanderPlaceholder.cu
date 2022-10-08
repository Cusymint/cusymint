#include "ExpanderPlaceholder.cuh"

#include "Symbol.cuh"
#include <fmt/core.h>

namespace Sym {
    DEFINE_SIMPLE_COMPRESS_REVERSE_TO(ExpanderPlaceholder)
    DEFINE_SIMPLE_COMPARE(ExpanderPlaceholder)
    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(ExpanderPlaceholder)

    __host__ __device__ ExpanderPlaceholder ExpanderPlaceholder::with_size(size_t size) {
        return {
            .type = Sym::Type::ExpanderPlaceholder,
            .size = size,
            .simplified = true,
        };
    }

    std::string ExpanderPlaceholder::to_string() const {
        return fmt::format("ExpanderPlaceholder({})", size);
    }

    std::string ExpanderPlaceholder::to_tex() const {
        return fmt::format(R"((\cdot)_{{ size={} }})", size);
    }
}
