#include "SubexpressionCandidate.cuh"

#include "Symbol.cuh"
#include "Symbol/Macros.cuh"
#include <fmt/core.h>

namespace Sym {
    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(SubexpressionCandidate)
    DEFINE_ARE_EQUAL(SubexpressionCandidate) {
        if (!(BASE_ARE_EQUAL(SubexpressionCandidate)) ||
            !(ONE_ARGUMENT_OP_ARE_EQUAL(SubexpressionCandidate))) {
            return false;
        }
        const auto& other_candidate = symbol->as<SubexpressionCandidate>();
        return vacancy_expression_idx == other_candidate.vacancy_expression_idx &&
               vacancy_idx == other_candidate.vacancy_idx &&
               subexpressions_left == other_candidate.subexpressions_left;
    }
    DEFINE_INVALID_COMPARE_TO(SubexpressionCandidate)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(SubexpressionCandidate)
    DEFINE_INVALID_DERIVATIVE(SubexpressionCandidate)

    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(SubexpressionCandidate)

    DEFINE_IS_FUNCTION_OF(SubexpressionCandidate) {
        return arg().is_function_of(expressions, expression_count);
    }

    [[nodiscard]] std::string SubexpressionCandidate::to_string() const {
        return fmt::format("SubexpressionCandidate{{uid={}, created_by={} ,({}, {}, {}), ({})}}",
                           uid, creator_uid, vacancy_expression_idx, vacancy_idx, subexpressions_left, arg().to_string());
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
        candidate.subexpressions_left = 0;
        return candidate_vec;
    }
}
