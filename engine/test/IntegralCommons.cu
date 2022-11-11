#include "IntegralCommons.cuh"

#include "Evaluation/Integrate.cuh"
#include "Parser/Parser.cuh"

namespace Test {
    namespace {
        testing::AssertionResult is_integral_solution(const std::string integral_str,
                                                      const std::string expected_result_str) {
            const auto integral = Sym::integral(parse_function(integral_str));
            const auto result = Sym::solve_integral(integral);

            auto expected_result = parse_function(expected_result_str);
            std::vector<Sym::Symbol> simplification_memory(Sym::EXPRESSION_MAX_SYMBOL_COUNT);
            expected_result.data()->simplify(simplification_memory.data());

            if (!result.has_value()) {
                return testing::AssertionFailure()
                       << "Tried to calculate the integral of:\n  " << integral_str
                       << "\n  but no result was found. The result should be:\n  "
                       << expected_result.data()->to_string();
            }

            if (Sym::Symbol::compare_trees(
                    result.value().data(), // NOLINT(bugprone-unchecked-optional-access)
                    expected_result.data())) {
                return testing::AssertionSuccess();
            }

            return testing::AssertionFailure()
                   << "Tried to calculate the integral of:\n  " << integral_str
                   << "\n  but got an unexpected result:\n  "
                   << result // NOLINT(bugprone-unchecked-optional-access)
                          .value()
                          .data()
                          ->to_string()
                   << " <- got\n  " << expected_result.data()->to_string() << " <- expected\n";
        }
    }

    void test_integral(const std::string integral_str, const std::string expected_result_str) {
        EXPECT_TRUE(is_integral_solution(integral_str, expected_result_str));
    }
}
