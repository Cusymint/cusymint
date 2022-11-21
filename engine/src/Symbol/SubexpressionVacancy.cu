#include "SubexpressionVacancy.cuh"

#include <fmt/format.h>

#include "Symbol/Macros.cuh"
#include "Symbol/Symbol.cuh"

namespace Sym {
    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(SubexpressionVacancy)
    DEFINE_SIMPLE_ARE_EQUAL(SubexpressionVacancy)
    DEFINE_INVALID_COMPARE_TO(SubexpressionVacancy)
    DEFINE_SIMPLE_COMPRESS_REVERSE_TO(SubexpressionVacancy)
    DEFINE_INVALID_IS_FUNCTION_OF(SubexpressionVacancy)
    DEFINE_NO_OP_PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE(SubexpressionVacancy)
    DEFINE_NO_OP_PUSH_CHILDREN_ONTO_STACK(SubexpressionVacancy)
    DEFINE_INVALID_DERIVATIVE(SubexpressionVacancy)
    DEFINE_SIMPLE_SEAL_WHOLE(SubexpressionVacancy);

    [[nodiscard]] std::string SubexpressionVacancy::to_string() const {
        std::string solved_info;

        if (is_solved == 1) {
            solved_info = fmt::format("solved by {}", solver_idx);
        }
        else {
            solved_info = fmt::format("candidate integrals: {}, candidate expressions: {}",
                                      candidate_integral_count, candidate_expression_count);
        }

        return fmt::format("SubexpressionVacancy{{ {} }}", solved_info);
    }

    [[nodiscard]] std::string SubexpressionVacancy::to_tex() const {
        return fmt::format("\\text{{ {} }}", to_string());
    }

    __host__ __device__ SubexpressionVacancy SubexpressionVacancy::for_single_integral() {
        SubexpressionVacancy vacancy = SubexpressionVacancy::create();
        vacancy.candidate_expression_count = 0;
        vacancy.candidate_integral_count = 1;
        vacancy.is_solved = 0;
        return vacancy;
    }

    std::vector<Symbol> single_integral_vacancy() {
        std::vector<Symbol> vacancy_vec(1);
        vacancy_vec.data()->init_from(SubexpressionVacancy::for_single_integral());
        return vacancy_vec;
    }
}
