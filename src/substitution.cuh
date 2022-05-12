#ifndef SUBSTITUTION_CUH
#define SUBSTITUTION_CUH

#include <vector>

#include "symbol_defs.cuh"

namespace Sym {
    DECLARE_SYMBOL(Substitution, false)
    static const size_t SUBSTITUTION_NAME_COUNT;
    static const char* const SUBSTITUTION_NAMES[];
    static std::string nth_substitution_name(const size_t n);
    size_t substitution_idx;
    size_t sub_substitution_count;
    std::string to_string() const;
    __host__ __device__ Symbol* next_substitution();
    __host__ __device__ const Symbol* next_substitution() const;
    END_DECLARE_SYMBOL(Substitution)

    std::vector<Symbol> substitute(const std::vector<Symbol>& integral,
                                   const std::vector<Symbol>& substitution);
}

#endif
