#include <cstddef>
#include <gtest/gtest.h>

#include <string>
#include <type_traits>
#include <vector>

#include <fmt/core.h>
#include <thrust/scan.h>

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
#include "Utils/DeviceArray.cuh"

#define KERNEL_TEST(_name) TEST(Kernels, _name)

using SymVector = std::vector<Sym::Symbol>;
using ExprVector = std::vector<SymVector>;
using StringVector = std::vector<std::string>;
using CheckVector = std::vector<Sym::KnownIntegral::Check>;
using IndexVector = std::vector<int>;

namespace Test {
    namespace {

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
            EXPECT_EQ(result_vector, expected_result);
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

        Sym::ExpressionArray<Sym::SubexpressionCandidate> from_cand_vector(ExprVector vector) {
            return Sym::ExpressionArray<Sym::SubexpressionCandidate>(
                vector, Sym::MAX_EXPRESSION_COUNT, Sym::EXPRESSION_MAX_SYMBOL_COUNT);
        }

        Sym::ExpressionArray<> from_vector(ExprVector vector) {
            return Sym::ExpressionArray<>(vector, Sym::MAX_EXPRESSION_COUNT,
                                          Sym::EXPRESSION_MAX_SYMBOL_COUNT);
        }

        Sym::ExpressionArray<Sym::SubexpressionCandidate>
        from_string_vector_with_candidate(StringVector vector) {
            return from_cand_vector(
                parse_strings_with_map(vector, Sym::first_expression_candidate));
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
            return candidate;
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
            result, parse_strings_with_map(solutions_vector, Sym::first_expression_candidate)));
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

        auto h_integrals = parse_strings_with_map(
            integrals_vector,
            Sym::
                first_expression_candidate); // from_string_vector_with_candidate(integrals_vector);
        for (int i = 0; i < h_integrals.size(); ++i) {
            h_integrals[i].data()->as<Sym::SubexpressionCandidate>().vacancy_expression_idx = i;
        }
        auto integrals = from_cand_vector(h_integrals);

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
            Sym::single_integral_vacancy(),
            nth_expression_candidate(5, Sym::solution(Sym::var()), 1),
            nth_expression_candidate(6, Sym::solution(Sym::var()), 1)};

        ExprVector expected_result = {
            vacancy_solved_by(1),
            nth_expression_candidate(0, vacancy_solved_by(2) + vacancy_solved_by(3)),
            nth_expression_candidate(1, vacancy_solved_by(5), 2),
            nth_expression_candidate(1, vacancy_solved_by(6), 3),
            Sym::single_integral_vacancy(),
            nth_expression_candidate(2, Sym::solution(Sym::var()), 1),
            nth_expression_candidate(3, Sym::solution(Sym::var()), 1),
            {},
            {},
            {}};

        Util::DeviceArray<uint32_t> removability(vacancy_tree.size(), true);
        auto expressions = from_vector(vacancy_tree);
        auto result = with_count(expressions.size());

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

        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        // EXPECT_EQ(result.to_vector(), expected_result);

        EXPECT_TRUE(are_expr_vectors_equal(result.to_vector(), expected_result));
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
        
        // TODO
    }

    // KERNEL_TEST(PropagateFailuresUpwards) {
    //     ExprVector vacancy_tree =
    //         {
    //             Sym::single_integral_vacancy(),
    //             nth_expression_candidate(0, Sym::single_integral_vacancy() + Sym::)
    //                 Sym::single_integral_vacancy(),
    //         }

    //     Sym::Kernel::propagate_failures_upwards(ExpressionArray<> expressions,
    //                                             Util::DeviceArray<uint32_t> failures)
    // }
}