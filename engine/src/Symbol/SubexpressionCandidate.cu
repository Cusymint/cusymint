#include "SubexpressionCandidate.cuh"

#include "Symbol.cuh"

namespace Sym {
    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(SubexpressionCandidate)
    DEFINE_SIMPLE_ONE_ARGUMETN_OP_COMPARE(SubexpressionCandidate)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(SubexpressionCandidate)

    DEFINE_SIMPLIFY_IN_PLACE(SubexpressionCandidate) { arg().simplify_in_place(help_space); }

    [[nodiscard]] std::string SubexpressionCandidate::to_string() const {
        return "SubexpressionCandidate{(" + std::to_string(vacancy_expression_idx) + ", " +
               std::to_string(vacancy_idx) + "), (" + arg().to_string() + ")}";
    }

    [[nodiscard]] std::string SubexpressionCandidate::to_tex() const {
        return fmt::format("\\text{{ {} }}", to_string());
    }

    __host__ __device__ void
    SubexpressionCandidate::copy_metadata_from(const SubexpressionCandidate& other) {
        vacancy_expression_idx = other.vacancy_expression_idx;
        vacancy_idx = other.vacancy_idx;
        subexpressions_left = other.subexpressions_left;
    }

    std::vector<Symbol> first_expression_candidate(const std::vector<Symbol>& child) {
        std::vector<Symbol> candidate_vec(1 + child.size());
        SubexpressionCandidate::create(child.data(), candidate_vec.data());
        auto& candidate = candidate_vec.data()->as<SubexpressionCandidate>();
        candidate.vacancy_expression_idx = 0;
        candidate.vacancy_idx = 0;
        candidate.subexpressions_left = 1;
        return candidate_vec;
    }
}
