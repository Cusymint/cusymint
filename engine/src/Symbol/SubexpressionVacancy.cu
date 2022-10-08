#include "SubexpressionVacancy.cuh"

#include <fmt/format.h>

#include "Symbol/Symbol.cuh"

namespace Sym {
    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(SubexpressionVacancy);
    DEFINE_SIMPLE_COMPARE(SubexpressionVacancy);
    DEFINE_SIMPLE_COMPRESS_REVERSE_TO(SubexpressionVacancy);

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

    std::vector<Symbol> single_integral_vacancy() {
        std::vector<Symbol> vacancy_vec(1);
        SubexpressionVacancy& vacancy =
            vacancy_vec.data()->init_from(SubexpressionVacancy::create());
        vacancy.candidate_expression_count = 0;
        vacancy.candidate_integral_count = 1;
        vacancy.is_solved = 0;
        return vacancy_vec;
    }
}
