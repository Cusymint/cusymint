#ifndef SOLUTION_CUH
#define SOLUTION_CUH

#include <vector>

#include "Macros.cuh"
#include "Substitution.cuh"

namespace Sym {
    DECLARE_SYMBOL(Solution, false)
    size_t substitution_count;
    size_t expression_offset;

    __host__ __device__ void seal_no_substitutions();
    __host__ __device__ void seal_single_substitution();
    __host__ __device__ void seal_substitutions(const size_t count, const size_t size);

    [[nodiscard]] __host__ __device__ const Substitution* first_substitution() const;
    [[nodiscard]] __host__ __device__ Substitution* first_substitution();

    [[nodiscard]] __host__ __device__ Symbol* expression();
    [[nodiscard]] __host__ __device__ const Symbol* expression() const;

    [[nodiscard]] __host__ __device__ size_t substitutions_size() const;

    [[nodiscard]] std::vector<Symbol> substitute_substitutions() const;

    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::string to_tex() const;

    END_DECLARE_SYMBOL(Solution)

    std::vector<Symbol> solution(const std::vector<Symbol>& arg);
}

#endif
