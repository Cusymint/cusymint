#include "IntegratorUtils.cuh"

namespace Test {
    using namespace Sym;

    std::string get_different_fields(ScanVector vec1, ScanVector vec2) {
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

    void test_known_integrals_correctly_checked(Util::DeviceArray<uint32_t> result,
                                                std::vector<IndexVector> index_vectors) {
        auto result_vector = result.to_vector();
        ScanVector expected_result(result.size());
        for (int i = 0; i < KnownIntegral::COUNT; ++i) {
            for (int j = 0; j < index_vectors.size(); ++j) {
                for (auto index : index_vectors[j]) {
                    if (i == index) {
                        expected_result[i * MAX_EXPRESSION_COUNT + j] = 1;
                    }
                }
            }
        }

        EXPECT_EQ(result_vector, expected_result)
            << get_different_fields(result_vector, expected_result);
    }

    void test_heuristics_correctly_checked(Util::DeviceArray<uint32_t> integral_result,
                                           Util::DeviceArray<uint32_t> expression_result,
                                           std::vector<HeuristicPairVector> heuristics) {
        ASSERT_EQ(integral_result.size(), expression_result.size());
        auto integral_result_vector = integral_result.to_vector();
        auto expression_result_vector = expression_result.to_vector();
        ScanVector expected_integral_result(integral_result.size());
        ScanVector expected_expression_result(expression_result.size());
        for (int i = 0; i < Heuristic::COUNT; ++i) {
            for (int j = 0; j < heuristics.size(); ++j) {
                for (auto heuristic : heuristics[j]) {
                    if (i == heuristic.first) {
                        expected_integral_result[i * MAX_EXPRESSION_COUNT + j] +=
                            heuristic.second.new_integrals;
                        expected_expression_result[i * MAX_EXPRESSION_COUNT + j] +=
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

    ExprVector parse_strings_with_map(StringVector& strings, SymVector (*map)(const SymVector&)) {
        ExprVector result;
        for (const auto& str : strings) {
            result.push_back(map(Parser::parse_function(str)));
        }

        return result;
    }

    std::string to_string_with_tab(const ExprVector& vec1) {
        std::string result = "[\n";
        for (const auto& expr : vec1) {
            if (expr.empty()) {
                result += "\t<empty>,\n";
            }
            else {
                result += "\t" + expr.data()->to_string() + ",\n";
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
                !Symbol::are_expressions_equal(*vec1[i].data(), *vec2[i].data())) {
                return testing::AssertionFailure() << failure_message(vec1, vec2);
            }
        }

        return testing::AssertionSuccess();
    }

    ExpressionArray<SubexpressionCandidate> from_string_vector_with_candidate(StringVector vector) {
        auto cand_vector = parse_strings_with_map(vector, first_expression_candidate);
        for (int i = 0; i < cand_vector.size(); ++i) {
            cand_vector[i].data()->as<SubexpressionCandidate>().vacancy_expression_idx = i;
        }

        return from_vector<SubexpressionCandidate>(cand_vector);
    }

    ExpressionArray<SubexpressionCandidate> with_count(const size_t count) {
        auto array = ExpressionArray<SubexpressionCandidate>(
            Integrator::INITIAL_ARRAYS_SYMBOLS_CAPACITY,
            Integrator::INITIAL_ARRAYS_EXPRESSIONS_CAPACITY);

        array.resize(count, Integrator::INITIAL_EXPRESSIONS_CAPACITY);

        return array;
    }

    SymVector vacancy_solved_by(size_t index) {
        SymVector vacancy = single_integral_vacancy();
        vacancy[0].as<SubexpressionVacancy>().is_solved = 1;
        vacancy[0].as<SubexpressionVacancy>().solver_idx = index;
        return vacancy;
    }

    SymVector vacancy(unsigned int candidate_integral_count,
                      unsigned int candidate_expression_count, int is_solved, size_t solver_idx) {
        SymVector vacancy = single_integral_vacancy();
        vacancy[0].as<SubexpressionVacancy>().candidate_integral_count = candidate_integral_count;
        vacancy[0].as<SubexpressionVacancy>().candidate_expression_count =
            candidate_expression_count;
        vacancy[0].as<SubexpressionVacancy>().is_solved = is_solved;
        vacancy[0].as<SubexpressionVacancy>().solver_idx = solver_idx;
        return vacancy;
    }

    SymVector failed_vacancy() { return vacancy(0, 0); }

    SymVector nth_expression_candidate(size_t n, const SymVector& child, size_t vacancy_idx) {
        SymVector candidate = first_expression_candidate(child);
        candidate[0].as<SubexpressionCandidate>().vacancy_expression_idx = n;
        candidate[0].as<SubexpressionCandidate>().vacancy_idx = vacancy_idx;
        size_t subexpressions_left = 0;
        for (auto symbol : child) {
            if (symbol.is(Type::SubexpressionVacancy) &&
                symbol.as<SubexpressionVacancy>().is_solved == 0) {
                ++subexpressions_left;
            }
        }

        candidate[0].as<SubexpressionCandidate>().subexpressions_left = subexpressions_left;
        return candidate;
    }

    SymVector nth_expression_candidate(size_t n, const std::string& child, size_t vacancy_idx) {
        return nth_expression_candidate(n, Parser::parse_function(child), vacancy_idx);
    }

    ExprVector get_expected_expression_vector(std::vector<HeuristicPairVector> heuristics_vector) {
        ExprVector result;
        for (const auto& heuristics : heuristics_vector) {
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
