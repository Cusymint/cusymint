#ifndef EXPANDER_PLACEHOLDER
#define EXPANDER_PLACEHOLDER

#include "Macros.cuh"

namespace Sym {
    DECLARE_SYMBOL(ExpanderPlaceholder, true)
    __host__ __device__ static ExpanderPlaceholder with_size(size_t size);
    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::string to_tex() const;
    DEFINE_IS_NOT_POLYNOMIAL
    END_DECLARE_SYMBOL(ExpanderPlaceholder)
}

#endif
