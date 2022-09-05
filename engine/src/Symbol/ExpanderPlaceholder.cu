#include "ExpanderPlaceholder.cuh"

#include "Symbol.cuh"

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
        return "ExpanderPlaceholder(" + std::to_string(size) + ")";
    }
}
