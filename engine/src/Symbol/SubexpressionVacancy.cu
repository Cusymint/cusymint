#include "SubexpressionVacancy.cuh"

namespace Sym {
    std::string SubexpressionVacancy::to_string() const {
        std::string solved_info;

        if (is_solved == 1) {
            solved_info = "solved by " + std::to_string(solver_idx);
        }
        else {
            solved_info = "candidate integrals: " + std::to_string(candidate_integral_count) +
                          ", candidate expressions: " + std::to_string(candidate_expression_count);
        }

        return "SubexpressionVacancy{" + solved_info + "}";
    }
}
