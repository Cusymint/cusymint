#ifndef SUBEXPRESSION_VACANCY_CUH
#define SUBEXPRESSION_VACANCY_CUH

#include "Macros.cuh"

namespace Sym {
    DECLARE_SYMBOL(SubexpressionVacancy, true)
    unsigned int candidate_integral_count;
    unsigned int candidate_expression_count;
    int is_solved; // Nie `bool`, bo nie ma implementacji atomicCAS dla `bool` w CUDA
    size_t solver_idx;
    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::string to_tex() const;
    __host__ __device__ static SubexpressionVacancy for_single_integral();
    END_DECLARE_SYMBOL(SubexpressionVacancy)

    std::vector<Symbol> single_integral_vacancy();
}

#endif
