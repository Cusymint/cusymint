#include "ExpanderPlaceholder.cuh"

#include "Symbol.cuh"
#include "Symbol/Macros.cuh"
#include <fmt/core.h>

namespace Sym {
    DEFINE_COMPRESS_REVERSE_TO(ExpanderPlaceholder) {
        Symbol* const new_destination = destination + additional_required_size;
        for (int i = 0; i < size; ++i) {
            symbol()->copy_single_to(new_destination + i);
        }
        new_destination->additional_required_size() = 0;
        return size + additional_required_size;
    }
    DEFINE_SIMPLE_COMPARE(ExpanderPlaceholder)
    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(ExpanderPlaceholder)
    DEFINE_INVALID_IS_FUNCTION_OF(ExpanderPlaceholder) // NOLINT
    DEFINE_NO_OP_PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE(ExpanderPlaceholder)

    __host__ __device__ ExpanderPlaceholder ExpanderPlaceholder::with_size(size_t size) {
        return {
            .type = Sym::Type::ExpanderPlaceholder,
            .size = size,
            .simplified = true,
            .additional_required_size = 0,
        };
    }

    std::string ExpanderPlaceholder::to_string() const {
        return fmt::format("ExpanderPlaceholder({})", size);
    }

    std::string ExpanderPlaceholder::to_tex() const {
        return fmt::format(R"((\cdot)_{{ size={} }})", size);
    }
}
