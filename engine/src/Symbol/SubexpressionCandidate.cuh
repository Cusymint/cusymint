#ifndef SUBEXPRESSION_CANDIDATE_CUH
#define SUBEXPRESSION_CANDIDATE_CUH

#include "Macros.cuh"

namespace Sym {
    DECLARE_SYMBOL(SubexpressionCandidate, false)
    ONE_ARGUMENT_OP_SYMBOL
    size_t vacancy_expression_idx;
    size_t vacancy_idx;
    std::string to_string() const;
    __host__ __device__ void copy_metadata_from(const SubexpressionCandidate& other);
    END_DECLARE_SYMBOL(SubexpressionCandidate)
}

#endif
