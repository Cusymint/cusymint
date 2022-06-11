#ifndef EXPANDER_PLACEHOLDER
#define EXPANDER_PLACEHOLDER

#include "symbol_defs.cuh"

namespace Sym {
    DECLARE_SYMBOL(ExpanderPlaceholder, true)
    __host__ __device__ static ExpanderPlaceholder with_size(size_t size);
    std::string to_string() const;
    END_DECLARE_SYMBOL(ExpanderPlaceholder)
}

#endif
