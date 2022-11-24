#include <cstddef>
#include <gtest/gtest.h>

#include <string>
#include <type_traits>
#include <vector>

#include <fmt/core.h>
#include <thrust/scan.h>

#include "Evaluation/Heuristic/Heuristic.cuh"
#include "Evaluation/Integrator.cuh"
#include "Evaluation/IntegratorKernels.cuh"
#include "Evaluation/KnownIntegral/KnownIntegral.cuh"
#include "Evaluation/KnownIntegral/SimpleExponent.cuh"
#include "Evaluation/KnownIntegral/SimpleSineCosine.cuh"
#include "Evaluation/KnownIntegral/SimpleVariable.cuh"
#include "Parser/Parser.cuh"
#include "Symbol/ExpressionArray.cuh"
#include "Symbol/Integral.cuh"
#include "Symbol/Solution.cuh"
#include "Symbol/SubexpressionCandidate.cuh"
#include "Symbol/SubexpressionVacancy.cuh"
#include "Symbol/Symbol.cuh"
#include "Symbol/SymbolType.cuh"
#include "Symbol/Variable.cuh"
#include "Utils/DeviceArray.cuh"
#include "Utils/Pair.cuh"

#define KERNEL_TEST(_name) TEST(Kernels, _name)

using SymVector = std::vector<Sym::Symbol>;
using ExprVector = std::vector<SymVector>;
using StringVector = std::vector<std::string>;
using CheckVector = std::vector<Sym::KnownIntegral::Check>;
using IndexVector = std::vector<int>;
using HeuristicPairVector = std::vector<Util::Pair<uint32_t, Sym::Heuristic::CheckResult>>;

namespace Test {
    namespace {

        std::string get_different_fields(std::vector<uint32_t> vec1, std::vector<uint32_t> vec2) {
            if (vec1.size() != vec2.size()) {
                return fmt::format("Vector sizes do not match: {} vs {}", vec1.size(), vec2.size());
            }

            std::string message = "Differences between vectors:\n";
            for (int i = 0; i < vec1.size(); ++i) {
                if (vec1[i] != vec2[i]) {
                    message += fmt::format("\tat {}: {} vs {}", i, vec1[i], vec2[i]);
                }
            }
            return message;
        }

        void test_correctly_checked(Util::DeviceArray<uint32_t> result,
                                    std::vector<IndexVector> index_vectors) {
            auto result_vector = result.to_vector();
            std::vector<uint32_t> expected_result(result.size());
            for (int i = 0; i < Sym::KnownIntegral::COUNT; ++i) {
                for (int j = 0; j < index_vectors.size(); ++j) {
                    for (auto index : index_vectors[j]) {
                        if (i == index) {
                            expected_result[i * Sym::MAX_EXPRESSION_COUNT + j] = 1;
                        }
                    }
                }
            }
            EXPECT_EQ(result_vector, expected_result)
                << get_different_fields(result_vector, expected_result);
        }

        void test_correctly_checked(Util::DeviceArray<uint32_t> integral_result,
                                    Util::DeviceArray<uint32_t> expression_result,
                                    std::vector<HeuristicPairVector> heuristics) {
            ASSERT_EQ(integral_result.size(), expression_result.size());
            auto integral_result_vector = integral_result.to_vector();
            auto expression_result_vector = expression_result.to_vector();
            std::vector<uint32_t> expected_integral_result(integral_result.size());
            std::vector<uint32_t> expected_expression_result(expression_result.size());
            for (int i = 0; i < Sym::Heuristic::COUNT; ++i) {
                for (int j = 0; j < heuristics.size(); ++j) {
                    for (auto heuristic : heuristics[j]) {
                        if (i == heuristic.first) {
                            expected_integral_result[i * Sym::MAX_EXPRESSION_COUNT + j] +=
                                heuristic.second.new_integrals;
                            expected_expression_result[i * Sym::MAX_EXPRESSION_COUNT + j] +=
                                heuristic.second.new_expressions;
                        }
                    }
                }
            }
            EXPECT_EQ(integral_result_vector, expected_integral_result)
                << get_different_fields(integral_result_vector, expected_integral_result);
            EXPECT_EQ(expression_result_vector, expected_expression_result)
                << get_different_fields(expression_result_vector, expected_expression_result);
        }

        ExprVector parse_strings_with_map(StringVector& strings,
                                          SymVector (*map)(const SymVector&)) {
            ExprVector result;
            for (auto str : strings) {
                result.push_back(map(Parser::parse_function(str)));
            }
            return result;
        }

        std::string to_string_with_tab(const ExprVector& vec1) {
            std::string result = "[\n";
            for (int i = 0; i < vec1.size(); ++i) {
                if (vec1[i].empty()) {
                    result += "\t<empty>,\n";
                }
                else {
                    result += "\t" + vec1[i].data()->to_string() + ",\n";
                }
            }
            return result + "]";
        }

        std::string failure_message(const ExprVector& vec1, const ExprVector& vec2) {
            return fmt::format("Unexpected result:\n{} <- got,\n{} <- expected",
                               to_string_with_tab(vec1), to_string_with_tab(vec2));
        }

        testing::AssertionResult are_expr_vectors_equal(const ExprVector& vec1,
                                                        const ExprVector& vec2) {
            if (vec1.size() != vec2.size()) {
                return testing::AssertionFailure() << failure_message(vec1, vec2);
            }
            for (int i = 0; i < vec1.size(); ++i) {
                if (vec1[i].empty() && vec2[i].empty()) {
                    continue;
                }
                if (vec1[i].empty() || vec2[i].empty() ||
                    !Sym::Symbol::are_expressions_equal(*vec1[i].data(), *vec2[i].data())) {
                    return testing::AssertionFailure() << failure_message(vec1, vec2);
                }
            }
            return testing::AssertionSuccess();
        }

        template <typename T = Sym::Symbol> Sym::ExpressionArray<T> from_vector(ExprVector vector) {
            return Sym::ExpressionArray<T>(vector, Sym::MAX_EXPRESSION_COUNT,
                                           Sym::EXPRESSION_MAX_SYMBOL_COUNT);
        }

        Sym::ExpressionArray<Sym::SubexpressionCandidate>
        from_string_vector_with_candidate(StringVector vector) {
            auto cand_vector = parse_strings_with_map(vector, Sym::first_expression_candidate);
            for (int i = 0; i < cand_vector.size(); ++i) {
                cand_vector[i].data()->as<Sym::SubexpressionCandidate>().vacancy_expression_idx = i;
            }

            return from_vector<Sym::SubexpressionCandidate>(cand_vector);
        }

        Sym::ExpressionArray<Sym::SubexpressionCandidate> with_count(const size_t count) {
            return Sym::ExpressionArray<Sym::SubexpressionCandidate>(
                Sym::EXPRESSION_MAX_SYMBOL_COUNT, Sym::MAX_EXPRESSION_COUNT, count);
        }

        SymVector vacancy_solved_by(size_t index) {
            SymVector vacancy = Sym::single_integral_vacancy();
            vacancy[0].as<Sym::SubexpressionVacancy>().is_solved = 1;
            vacancy[0].as<Sym::SubexpressionVacancy>().solver_idx = index;
            return vacancy;
        }

        SymVector vacancy(unsigned int candidate_integral_count,
                          unsigned int candidate_expression_count, int is_solved = 0,
                          size_t solver_idx = 0) {
            SymVector vacancy = Sym::single_integral_vacancy();
            vacancy[0].as<Sym::SubexpressionVacancy>().candidate_integral_count =
                candidate_integral_count;
            vacancy[0].as<Sym::SubexpressionVacancy>().candidate_expression_count =
                candidate_expression_count;
            vacancy[0].as<Sym::SubexpressionVacancy>().is_solved = is_solved;
            vacancy[0].as<Sym::SubexpressionVacancy>().solver_idx = solver_idx;
            return vacancy;
        }

        SymVector nth_expression_candidate(size_t n, const SymVector& child,
                                           size_t vacancy_idx = 0) {
            SymVector candidate = Sym::first_expression_candidate(child);
            candidate[0].as<Sym::SubexpressionCandidate>().vacancy_expression_idx = n;
            candidate[0].as<Sym::SubexpressionCandidate>().vacancy_idx = vacancy_idx;
            size_t subexpressions_left = 0;
            for (auto symbol : child) {
                if (symbol.is(Sym::Type::SubexpressionVacancy) &&
                    symbol.as<Sym::SubexpressionVacancy>().is_solved == 0) {
                    ++subexpressions_left;
                }
            }
            candidate[0].as<Sym::SubexpressionCandidate>().subexpressions_left =
                subexpressions_left;
            return candidate;
        }

        ExprVector
        get_expected_expression_vector(std::vector<HeuristicPairVector> heuristics_vector) {
            ExprVector result;
            for (auto heuristics : heuristics_vector) {
                size_t expression_count = 0;
                size_t integral_count = 0;
                for (auto heuristic : heuristics) {
                    if (heuristic.second.new_expressions == 0) {
                        integral_count += heuristic.second.new_integrals;
                    }
                    else {
                        expression_count += heuristic.second.new_expressions;
                    }
                }
                result.push_back(vacancy(integral_count, expression_count));
            }
            return result;
        }
    }

    KERNEL_TEST(Simplify) {
        StringVector expressions_vector = {
            "(sin(x)+cos(x)-20^e^x+pi)/(sin(x)+cos(x)-20^e^x+pi)",
            "10^(sin(x)*x/ln(10)*pi)",
            "(9+2*x^2+x^3)/(3+x+5*x^2+10*x^3+x^6)",
            "1/(1/(1/(1/(x))))",
            "2*5*7*x^5/(--(14*x^2))",
        }; //"3*(2*x+4*(10*x+2)+5)+1"};
        StringVector solutions_vector = {
            "1", "e^(pi*x*sin(x))", "(9+2*x^2+x^3)/(3+x+5*x^2+10*x^3+x^6)", "x", "5*x^3",
        }; // "40+126*x"};
        Sym::ExpressionArray<Sym::SubexpressionCandidate> expressions =
            from_string_vector_with_candidate(expressions_vector);
        Sym::ExpressionArray<Sym::SubexpressionCandidate> destination =
            with_count(expressions.size());
        Sym::ExpressionArray<Sym::SubexpressionCandidate> help_spaces =
            with_count(expressions.size());

        Sym::Kernel::simplify<<<Sym::Integrator::BLOCK_COUNT, Sym::Integrator::BLOCK_SIZE>>>(
            expressions, destination, help_spaces);

        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        ExprVector result = destination.to_vector();

        EXPECT_TRUE(are_expr_vectors_equal(
            result, from_string_vector_with_candidate(solutions_vector).to_vector()));
    }

    KERNEL_TEST(CheckForKnownIntegrals) {
        using namespace Sym::KnownIntegral;

        StringVector integrals_vector = {"int sin(x)", "int e^x", "int x", "int e^x + 1"};
        std::vector<IndexVector> check_vectors = {{3}, {2}, {0}, {}};

        auto integrals = from_string_vector_with_candidate(integrals_vector);
        Util::DeviceArray<uint32_t> applicability(COUNT * Sym::MAX_EXPRESSION_COUNT);

        Sym::Kernel::check_for_known_integrals<<<Sym::Integrator::BLOCK_COUNT,
                                                 Sym::Integrator::BLOCK_SIZE>>>(integrals,
                                                                                applicability);

        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        test_correctly_checked(applicability, check_vectors);
    }

    KERNEL_TEST(ApplyKnownIntegrals) {
        using namespace Sym::KnownIntegral;

        StringVector integrals_vector = {"int sin(x)", "int e^x", "int x", "int e^x + 1"};
        ExprVector expected_results = {
            vacancy_solved_by(6),
            vacancy_solved_by(5),
            vacancy_solved_by(4),
            Sym::single_integral_vacancy(),
            nth_expression_candidate(2, Sym::solution(Sym::num(0.5) * (Sym::var() ^ Sym::num(2)))),
            nth_expression_candidate(1, Sym::solution(Sym::e() ^ Sym::var())),
            nth_expression_candidate(0, Sym::solution(-Sym::cos(Sym::var())))};

        auto integrals = from_string_vector_with_candidate(integrals_vector);

        Util::DeviceArray<uint32_t> applicability(COUNT * Sym::MAX_EXPRESSION_COUNT);

        Sym::Kernel::check_for_known_integrals<<<Sym::Integrator::BLOCK_COUNT,
                                                 Sym::Integrator::BLOCK_SIZE>>>(integrals,
                                                                                applicability);

        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        cudaDeviceSynchronize();

        thrust::inclusive_scan(thrust::device, applicability.begin(), applicability.end(),
                               applicability.data());
        cudaDeviceSynchronize();

        auto expressions =
            from_vector({Sym::single_integral_vacancy(), Sym::single_integral_vacancy(),
                         Sym::single_integral_vacancy(), Sym::single_integral_vacancy()});
        auto help_spaces = with_count(integrals.size());

        Sym::Kernel::
            apply_known_integrals<<<Sym::Integrator::BLOCK_COUNT, Sym::Integrator::BLOCK_SIZE>>>(
                integrals, expressions, help_spaces, applicability);

        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        expressions.increment_size_from_device(applicability.last());

        EXPECT_TRUE(are_expr_vectors_equal(expressions.to_vector(), expected_results));
    }

    KERNEL_TEST(PropagateSolvedSubexpressions) {
        ExprVector vacancy_tree = {
            vacancy(9, 6),
            nth_expression_candidate(0, vacancy(5, 3) + vacancy(4, 2)),
            nth_expression_candidate(1, Sym::sin(vacancy(2, 1)) * vacancy(3, 1), 2),
            nth_expression_candidate(
                2, Sym::single_integral_vacancy() + Sym::single_integral_vacancy(), 3),
            nth_expression_candidate(
                2,
                Sym::single_integral_vacancy() +
                    (Sym::single_integral_vacancy() ^ Sym::single_integral_vacancy()),
                4),
            nth_expression_candidate(1, -vacancy(3, 2), 3),
            nth_expression_candidate(5, vacancy_solved_by(10) + vacancy(2, 1), 2),
            nth_expression_candidate(6, vacancy_solved_by(9) * vacancy_solved_by(8), 3)};

        ExprVector result_tree = {
            vacancy(9, 6),
            nth_expression_candidate(0, vacancy(5, 3) + vacancy_solved_by(5)),
            nth_expression_candidate(1, Sym::sin(vacancy(2, 1)) * vacancy(3, 1), 2),
            nth_expression_candidate(
                2, Sym::single_integral_vacancy() + Sym::single_integral_vacancy(), 3),
            nth_expression_candidate(
                2,
                Sym::single_integral_vacancy() +
                    (Sym::single_integral_vacancy() ^ Sym::single_integral_vacancy()),
                4),
            nth_expression_candidate(1, -vacancy_solved_by(6), 3),
            nth_expression_candidate(5, vacancy_solved_by(10) + vacancy_solved_by(7), 2),
            nth_expression_candidate(6, vacancy_solved_by(9) * vacancy_solved_by(8), 3)};

        auto expressions = from_vector(vacancy_tree);

        Sym::Kernel::propagate_solved_subexpressions<<<Sym::Integrator::BLOCK_COUNT,
                                                       Sym::Integrator::BLOCK_SIZE>>>(expressions);

        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        EXPECT_TRUE(are_expr_vectors_equal(expressions.to_vector(), result_tree));
    }

    KERNEL_TEST(FindRedundantExpressions) {
        ExprVector vacancy_tree = {
            vacancy_solved_by(1),
            nth_expression_candidate(0, vacancy_solved_by(5) + vacancy_solved_by(6)),
            nth_expression_candidate(0, Sym::single_integral_vacancy() * vacancy_solved_by(7)),
            nth_expression_candidate(1, vacancy_solved_by(100) + vacancy_solved_by(100), 2),
            nth_expression_candidate(1, Sym::single_integral_vacancy() + vacancy_solved_by(100), 3),
            nth_expression_candidate(1, vacancy_solved_by(200), 2),
            nth_expression_candidate(1, vacancy_solved_by(300), 3),
            Sym::single_integral_vacancy()};

        std::vector<uint32_t> expected_result = {1, 1, 0, 0, 0, 1, 1, 1};
        Util::DeviceArray<uint32_t> removability(vacancy_tree.size(), true);

        Sym::Kernel::find_redundand_expressions<<<Sym::Integrator::BLOCK_COUNT,
                                                  Sym::Integrator::BLOCK_SIZE>>>(
            from_vector(vacancy_tree), removability);

        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        EXPECT_EQ(removability.to_vector(), expected_result);
    }

    KERNEL_TEST(FindRedundantIntegrals) {
        ExprVector integrals_tree = {nth_expression_candidate(2, Sym::integral(Sym::var()), 2),
                                     nth_expression_candidate(7, Sym::integral(Sym::var()), 0),
                                     nth_expression_candidate(4, Sym::integral(Sym::var()), 2)};

        ExprVector vacancy_tree = {
            vacancy_solved_by(1),
            nth_expression_candidate(0, vacancy_solved_by(5) + vacancy_solved_by(6)),
            nth_expression_candidate(0, Sym::single_integral_vacancy() * vacancy_solved_by(7)),
            nth_expression_candidate(1, vacancy_solved_by(100) + vacancy_solved_by(100), 2),
            nth_expression_candidate(1, Sym::single_integral_vacancy() + vacancy_solved_by(100), 3),
            nth_expression_candidate(1, vacancy_solved_by(200), 2),
            nth_expression_candidate(1, vacancy_solved_by(300), 3),
            Sym::single_integral_vacancy()};

        std::vector<uint32_t> expected_result = {0, 1, 0};

        Util::DeviceArray<uint32_t> removability(vacancy_tree.size(), true);
        Util::DeviceArray<uint32_t> integral_removability(integrals_tree.size(), true);

        Sym::Kernel::find_redundand_expressions<<<Sym::Integrator::BLOCK_COUNT,
                                                  Sym::Integrator::BLOCK_SIZE>>>(
            from_vector(vacancy_tree), removability);

        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        Sym::Kernel::
            find_redundand_integrals<<<Sym::Integrator::BLOCK_COUNT, Sym::Integrator::BLOCK_SIZE>>>(
                from_vector(integrals_tree), from_vector(vacancy_tree), removability,
                integral_removability);

        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        EXPECT_EQ(integral_removability.to_vector(), expected_result);
    }

    KERNEL_TEST(RemoveExpressions) {
        ExprVector vacancy_tree = {
            vacancy_solved_by(1),
            nth_expression_candidate(0, vacancy_solved_by(5) + vacancy_solved_by(6)),
            nth_expression_candidate(0, Sym::single_integral_vacancy() * vacancy_solved_by(7)),
            nth_expression_candidate(1, vacancy_solved_by(10) + vacancy_solved_by(11), 2),
            nth_expression_candidate(1, Sym::single_integral_vacancy() + vacancy_solved_by(12), 3),
            nth_expression_candidate(1, vacancy_solved_by(8), 2),
            nth_expression_candidate(1, vacancy_solved_by(9), 3),
            vacancy(3, 1),
            nth_expression_candidate(5, Sym::solution(Sym::var()), 1),
            nth_expression_candidate(6, Sym::solution(Sym::var()), 1),
            nth_expression_candidate(7, Sym::single_integral_vacancy())};

        ExprVector expected_result = {
            vacancy_solved_by(1),
            nth_expression_candidate(0, vacancy_solved_by(2) + vacancy_solved_by(3)),
            nth_expression_candidate(1, vacancy_solved_by(5), 2),
            nth_expression_candidate(1, vacancy_solved_by(6), 3),
            vacancy(3, 1),
            nth_expression_candidate(2, Sym::solution(Sym::var()), 1),
            nth_expression_candidate(3, Sym::solution(Sym::var()), 1),
            nth_expression_candidate(4, Sym::single_integral_vacancy())};

        ExprVector expected_result_zeroed(expected_result);
        expected_result_zeroed[4].data()->as<Sym::SubexpressionVacancy>().candidate_integral_count = 0;
        expected_result_zeroed[7].data()->as<Sym::SubexpressionCandidate>().arg().as<Sym::SubexpressionVacancy>().candidate_integral_count = 0;

        Util::DeviceArray<uint32_t> removability(vacancy_tree.size(), true);
        auto expressions = from_vector(vacancy_tree);
        auto result = with_count(expressions.size());
        auto result_zeroed = with_count(expressions.size());

        Sym::Kernel::find_redundand_expressions<<<Sym::Integrator::BLOCK_COUNT,
                                                  Sym::Integrator::BLOCK_SIZE>>>(expressions,
                                                                                 removability);

        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        thrust::inclusive_scan(thrust::device, removability.begin(), removability.end(),
                               removability.data());

        cudaDeviceSynchronize();

        Sym::Kernel::
            remove_expressions<<<Sym::Integrator::BLOCK_COUNT, Sym::Integrator::BLOCK_SIZE>>>(
                expressions, removability, result);

        Sym::Kernel::
            remove_expressions<true><<<Sym::Integrator::BLOCK_COUNT, Sym::Integrator::BLOCK_SIZE>>>(
                expressions, removability, result_zeroed);

        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        result.resize(removability.to_cpu(removability.size() - 1));
        result_zeroed.resize(removability.to_cpu(removability.size() - 1));

        EXPECT_TRUE(are_expr_vectors_equal(result.to_vector(), expected_result));
        EXPECT_TRUE(are_expr_vectors_equal(result_zeroed.to_vector(), expected_result_zeroed));
    }

    KERNEL_TEST(RemoveIntegrals) {
        ExprVector vacancy_tree = {
            vacancy_solved_by(1),
            nth_expression_candidate(0, vacancy_solved_by(5) + vacancy_solved_by(6)),
            nth_expression_candidate(0, Sym::single_integral_vacancy() * vacancy_solved_by(7)),
            nth_expression_candidate(1, vacancy_solved_by(10) + vacancy_solved_by(11), 2),
            nth_expression_candidate(1, Sym::single_integral_vacancy() + vacancy_solved_by(12), 3),
            nth_expression_candidate(1, vacancy_solved_by(8), 2),
            nth_expression_candidate(1, vacancy_solved_by(9), 3),
            Sym::single_integral_vacancy(),
            nth_expression_candidate(5, Sym::solution(Sym::var()), 1),
            nth_expression_candidate(6, Sym::solution(Sym::var()), 1)};

        ExprVector integral_vector = {nth_expression_candidate(2, Sym::integral(Sym::var()), 2),
                                      nth_expression_candidate(7, Sym::integral(Sym::e())),
                                      nth_expression_candidate(4, Sym::integral(Sym::pi()), 2)};

        std::vector<uint32_t> expressions_removability_scan_vector = {1, 2, 2, 2, 2, 3, 4, 5, 6, 7};
        std::vector<uint32_t> integral_removability_scan_vector = {0, 1, 1};

        ExprVector expected_result = {nth_expression_candidate(
            expressions_removability_scan_vector[7] - 1, Sym::integral(Sym::e()))};

        Util::DeviceArray<uint32_t> expressions_removability_scan(
            expressions_removability_scan_vector);
        Util::DeviceArray<uint32_t> integral_removability_scan(integral_removability_scan_vector);

        auto result = with_count(integral_vector.size());

        Sym::Kernel::
            remove_integrals<<<Sym::Integrator::BLOCK_COUNT, Sym::Integrator::BLOCK_SIZE>>>(
                from_vector<Sym::SubexpressionCandidate>(integral_vector),
                integral_removability_scan, expressions_removability_scan, result);

        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        result.resize(
            integral_removability_scan_vector[integral_removability_scan_vector.size() - 1]);

        EXPECT_TRUE(are_expr_vectors_equal(result.to_vector(), expected_result));
    }

    KERNEL_TEST(CheckHeuristicsApplicability) {
        using namespace Sym::Heuristic;

        StringVector integrals_vector = {"int sin(x)+cos(x) dx", "int e^x*e^e^x dx",
                                         "int 23*c*x dx",        "int x+2 dx",
                                         "int 2*tan(0.5*x) dx",  "int e^x^2 dx"};

        ExprVector expressions_vector = {vacancy(0, 0), vacancy(0, 0), vacancy(0, 0),
                                         vacancy(0, 0), vacancy(0, 0), vacancy(0, 0)};

        std::vector<HeuristicPairVector> expected_heuristics = {{{1, {2, 1}}, {2, {1, 0}}},
                                                                {{0, {1, 0}}},
                                                                {{3, {1, 1}}},
                                                                {{1, {2, 1}}},
                                                                {{2, {1, 0}}, {3, {1, 1}}},
                                                                {}};

        ExprVector expected_expressions_vector =
            get_expected_expression_vector(expected_heuristics);

        auto expressions = from_vector(expressions_vector);
        Util::DeviceArray<uint32_t> new_integrals_flags(COUNT * Sym::MAX_EXPRESSION_COUNT);
        Util::DeviceArray<uint32_t> new_expressions_flags(COUNT * Sym::MAX_EXPRESSION_COUNT);

        Sym::Kernel::check_heuristics_applicability<<<Sym::Integrator::BLOCK_COUNT,
                                                      Sym::Integrator::BLOCK_SIZE>>>(
            from_string_vector_with_candidate(integrals_vector), expressions, new_integrals_flags,
            new_expressions_flags);

        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        EXPECT_TRUE(are_expr_vectors_equal(expressions.to_vector(), expected_expressions_vector));
        test_correctly_checked(new_integrals_flags, new_expressions_flags, expected_heuristics);
    }

    KERNEL_TEST(ApplyHeuristics) {
        using namespace Sym::Heuristic;

        StringVector integrals_vector = {"int sin(x)+cos(x) dx", "int e^x*e^e^x dx",
                                         "int 23*c*x dx",        "int x+2 dx",
                                         "int 2*tan(0.5*x) dx",  "int e^x^2 dx"};

        ExprVector expressions_vector = {vacancy(0, 0), vacancy(0, 0), vacancy(0, 0),
                                         vacancy(0, 0), vacancy(0, 0), vacancy(0, 0)};

        std::vector<HeuristicPairVector> expected_heuristics = {{{1, {2, 1}}, {2, {1, 0}}},
                                                                {{0, {1, 0}}},
                                                                {{3, {1, 1}}},
                                                                {{1, {2, 1}}},
                                                                {{2, {1, 0}}, {3, {1, 1}}},
                                                                {}};

        ExprVector expected_expression_vector = get_expected_expression_vector(expected_heuristics);

        auto h_integrals =
            parse_strings_with_map(integrals_vector, Sym::first_expression_candidate);
        for (int i = 0; i < h_integrals.size(); ++i) {
            h_integrals[i].data()->as<Sym::SubexpressionCandidate>().vacancy_expression_idx = i;
        }

        // e tower with substitution
        SymVector e_tower_integral =
            Sym::integral((Sym::e() ^ Sym::var()) * (Sym::e() ^ (Sym::e() ^ Sym::var())));
        SymVector int_with_subs(e_tower_integral.size() * 2);
        SymVector e_to_x = Sym::e() ^ Sym::var();
        SymVector var = Sym::var();
        Util::Pair<const Sym::Symbol*, const Sym::Symbol*> pairs[] = {{e_to_x.data(), var.data()}};
        e_tower_integral.data()->as<Sym::Integral>().integrate_by_substitution_with_derivative(
            pairs, 1, *Sym::var().data(), *int_with_subs.data());
        int_with_subs.resize(int_with_subs.data()->size());

        // trigs with substitution
        StringVector trig_substitutions_strings = {
            "tan(0.5*x)", "x",           "sin(x)", "2*x/(1+x^2)", "cos(x)",   "(1-x^2)/(1+x^2)",
            "tan(x)",     "2*x/(1-x^2)", "cot(x)", "(1-x^2)/2*x", "(1+x^2)/2"};
        auto trig_substitutions =
            parse_strings_with_map(trig_substitutions_strings, [](auto& map) { return map; });
        Util::Pair<const Sym::Symbol*, const Sym::Symbol*> trig_pairs[] = {
            {trig_substitutions[0].data(), trig_substitutions[1].data()},
            {trig_substitutions[2].data(), trig_substitutions[3].data()},
            {trig_substitutions[4].data(), trig_substitutions[5].data()},
            {trig_substitutions[6].data(), trig_substitutions[7].data()},
            {trig_substitutions[8].data(), trig_substitutions[9].data()}};
        SymVector trig1_with_subs(integrals_vector[0].size() * 6);
        SymVector trig2_with_subs(integrals_vector[4].size() * 6);
        h_integrals[0]
            .data()
            ->child()
            ->as<Sym::Integral>()
            .integrate_by_substitution_with_derivative(
                trig_pairs, 5, *trig_substitutions[10].data(), *trig1_with_subs.data());
        h_integrals[4]
            .data()
            ->child()
            ->as<Sym::Integral>()
            .integrate_by_substitution_with_derivative(
                trig_pairs, 5, *trig_substitutions[10].data(), *trig2_with_subs.data());
        trig1_with_subs.resize(trig1_with_subs.data()->size());
        trig2_with_subs.resize(trig2_with_subs.data()->size());

        expected_expression_vector.insert(
            expected_expression_vector.end(),
            {
                nth_expression_candidate(0, Sym::single_integral_vacancy() +
                                                Sym::single_integral_vacancy()),
                nth_expression_candidate(3, Sym::single_integral_vacancy() +
                                                Sym::single_integral_vacancy()),
                nth_expression_candidate(2, Sym::single_integral_vacancy() * Sym::cnst("c") *
                                                Sym::num(23)),
                nth_expression_candidate(4, Sym::single_integral_vacancy() * Sym::num(2)),
            });

        ExprVector expected_integral_vector = {
            nth_expression_candidate(1, int_with_subs),
            nth_expression_candidate(6, Sym::integral(Sym::sin(Sym::var())), 2),
            nth_expression_candidate(6, Sym::integral(Sym::cos(Sym::var())), 3),
            nth_expression_candidate(7, Sym::integral(Sym::var()), 2),
            nth_expression_candidate(7, Sym::integral(Sym::num(2)), 3),
            nth_expression_candidate(0, trig1_with_subs),
            nth_expression_candidate(4, trig2_with_subs),
            nth_expression_candidate(8, Sym::integral(Sym::var()), 3),
            nth_expression_candidate(9, Sym::integral(Sym::tan(Sym::num(0.5) * Sym::var())), 2),
        };

        auto integrals = from_vector<Sym::SubexpressionCandidate>(h_integrals);
        auto expressions = from_vector(expressions_vector);

        Util::DeviceArray<uint32_t> new_integrals_flags(COUNT * Sym::MAX_EXPRESSION_COUNT);
        Util::DeviceArray<uint32_t> new_expressions_flags(COUNT * Sym::MAX_EXPRESSION_COUNT);

        Sym::Kernel::check_heuristics_applicability<<<Sym::Integrator::BLOCK_COUNT,
                                                      Sym::Integrator::BLOCK_SIZE>>>(
            integrals, expressions, new_integrals_flags, new_expressions_flags);

        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        thrust::inclusive_scan(thrust::device, new_integrals_flags.begin(),
                               new_integrals_flags.end(), new_integrals_flags.data());
        thrust::inclusive_scan(thrust::device, new_expressions_flags.begin(),
                               new_expressions_flags.end(), new_expressions_flags.data());
        cudaDeviceSynchronize();

        auto integrals_destinations = with_count(2 * integrals_vector.size());
        auto help_spaces = with_count(2 * integrals_vector.size());

        Sym::Kernel::
            apply_heuristics<<<Sym::Integrator::BLOCK_COUNT, Sym::Integrator::BLOCK_SIZE>>>(
                integrals, integrals_destinations, expressions, help_spaces, new_integrals_flags,
                new_expressions_flags);

        integrals_destinations.resize(new_integrals_flags.to_cpu(new_integrals_flags.size() - 1));
        expressions.resize(expressions.size() +
                           new_expressions_flags.to_cpu(new_expressions_flags.size() - 1));

        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        EXPECT_TRUE(
            are_expr_vectors_equal(integrals_destinations.to_vector(), expected_integral_vector));
        EXPECT_TRUE(are_expr_vectors_equal(expressions.to_vector(), expected_expression_vector));
    }

    KERNEL_TEST(PropagateFailuresUpwards) {
        ExprVector vacancy_tree = {
            vacancy(0, 2) /*child failed but another subexpression remains*/,
            nth_expression_candidate(0, vacancy(0, 1) + vacancy(0, 1)),
            nth_expression_candidate(1, Sym::sin(vacancy(0, 1)) * vacancy(0, 1), 2),
            nth_expression_candidate(
                2, Sym::single_integral_vacancy() + Sym::single_integral_vacancy(), 3),
            nth_expression_candidate(
                2,
                Sym::single_integral_vacancy() +
                    (Sym::single_integral_vacancy() ^ Sym::single_integral_vacancy()),
                4),
            nth_expression_candidate(1, -vacancy(0, 1), 3),
            nth_expression_candidate(5, Sym::single_integral_vacancy() + vacancy(0, 1), 2),
            nth_expression_candidate(6, vacancy(0, 0) /*failed*/ *
            Sym::single_integral_vacancy(), 3),
            nth_expression_candidate(0, vacancy(1, 1)) /*child failed but one integral remains*/,
            nth_expression_candidate(8, vacancy(0,0) /*failed*/, 1)};

        std::vector<uint32_t> expected_failures_vector = {1,0,1,1,1,0,0,0, 1, 0};

        auto expressions = from_vector(vacancy_tree);
        Util::DeviceArray<uint32_t> failures(vacancy_tree.size());
        failures.set_mem(1);

        Sym::Kernel::propagate_failures_upwards<<<Sym::Integrator::BLOCK_COUNT,
                                                     Sym::Integrator::BLOCK_SIZE>>>(expressions,
                                                     failures);
                                            
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        EXPECT_EQ(failures.to_vector(), expected_failures_vector);
    }
}