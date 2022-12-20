#ifndef INTEGRATOR_UTILS_H
#define INTEGRATOR_UTILS_H

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "Evaluation/Heuristic/Heuristic.cuh"
#include "Evaluation/Integrator.cuh"
#include "Evaluation/IntegratorKernels.cuh"
#include "Evaluation/KnownIntegral/KnownIntegral.cuh"
#include "Evaluation/KnownIntegral/SimpleExponent.cuh"
#include "Evaluation/KnownIntegral/SimpleSineCosine.cuh"
#include "Evaluation/KnownIntegral/SimpleVariable.cuh"
#include "Evaluation/Status.cuh"
#include "Parser/Parser.cuh"
#include "Symbol/ExpressionArray.cuh"
#include "Symbol/Integral.cuh"
#include "Symbol/Macros.cuh"
#include "Symbol/Solution.cuh"
#include "Symbol/SubexpressionCandidate.cuh"
#include "Symbol/SubexpressionVacancy.cuh"
#include "Symbol/Symbol.cuh"
#include "Symbol/SymbolType.cuh"
#include "Symbol/Variable.cuh"
#include "Utils/DeviceArray.cuh"
#include "Utils/Pair.cuh"

namespace Test {
    using SymVector = std::vector<Sym::Symbol>;
    using ExprVector = std::vector<SymVector>;
    using StringVector = std::vector<std::string>;
    using CheckVector = std::vector<Sym::KnownIntegral::Check>;
    using IndexVector = std::vector<int>;
    using HeuristicPairVector = std::vector<Util::Pair<uint32_t, Sym::Heuristic::CheckResult>>;
    using ScanVector = std::vector<uint32_t>;
    using EvalStatusVector = std::vector<Sym::EvaluationStatus>;

    Sym::SymbolIterator iterator_from_vector(SymVector& vector);

    std::string get_different_fields(ScanVector vec1, ScanVector vec2);

    void test_known_integrals_correctly_checked(Util::DeviceArray<uint32_t> result,
                                                std::vector<IndexVector> index_vectors,
                                                size_t integral_count);

    void test_heuristics_correctly_checked(Util::DeviceArray<uint32_t> integral_result,
                                           Util::DeviceArray<uint32_t> expression_result,
                                           std::vector<HeuristicPairVector> heuristics);

    ExprVector parse_strings_with_map(StringVector& strings, SymVector (*map)(const SymVector&));

    std::string to_string_with_tab(const ExprVector& vec1);

    std::string failure_message(const ExprVector& vec1, const ExprVector& vec2);

    testing::AssertionResult are_expr_vectors_equal(const ExprVector& vec1, const ExprVector& vec2);

    testing::AssertionResult are_expr_vectors_equal_with_statuses(const ExprVector& vec1, const EvalStatusVector& statuses1, const ExprVector& vec2, const EvalStatusVector& statuses2);

    Sym::ExpressionArray<Sym::SubexpressionCandidate>
    from_string_vector_with_candidate(StringVector vector);

    Sym::ExpressionArray<Sym::SubexpressionCandidate> with_count(const size_t count);

    SymVector vacancy_solved_by(size_t index);

    SymVector vacancy(unsigned int candidate_integral_count,
                      unsigned int candidate_expression_count, int is_solved = 0,
                      size_t solver_idx = 0);

    SymVector failed_vacancy();

    SymVector nth_expression_candidate(size_t n, const SymVector& child, size_t vacancy_idx = 0);

    SymVector nth_expression_candidate(size_t n, const std::string& child, size_t vacancy_idx = 0);

    SymVector nth_expression_candidate_with_uid(size_t uid, size_t n, const SymVector& child, size_t vacancy_idx = 0);

    SymVector nth_expression_candidate_with_uid(size_t uid, size_t n, const std::string& child, size_t vacancy_idx = 0);

    SymVector nth_expression_candidate_with_uids(size_t uid, size_t creator_uid, size_t n, const SymVector& child, size_t vacancy_idx = 0);

    ExprVector get_expected_expression_vector(std::vector<HeuristicPairVector> heuristics_vector);
}

#endif
