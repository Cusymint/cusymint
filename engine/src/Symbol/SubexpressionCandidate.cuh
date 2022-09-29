#ifndef SUBEXPRESSION_CANDIDATE_CUH
#define SUBEXPRESSION_CANDIDATE_CUH

#include "Macros.cuh"

namespace Sym {
    DECLARE_SYMBOL(SubexpressionCandidate, false)
    ONE_ARGUMENT_OP_SYMBOL
    size_t vacancy_expression_idx;
    size_t vacancy_idx;
    // Ile w w tym kandydacie pozostało nierozwiązanych podwyrażeń. Powinno być równe zero jeśli
    // znajduje się w tablicy całek.
    // Typ to nie `size_t`, bo `atomicSub` nie jest dla niego zdefiniowane
    unsigned int subexpressions_left;
    std::string to_string() const;
    __host__ __device__ void copy_metadata_from(const SubexpressionCandidate& other);
    END_DECLARE_SYMBOL(SubexpressionCandidate)
}

#endif
