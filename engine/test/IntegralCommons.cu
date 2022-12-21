#include "IntegralCommons.cuh"

#include <vector>

#include "Simplify.cuh"
#include "Evaluation/Integrator.cuh"
#include "Parser/Parser.cuh"
#include "Symbol/Macros.cuh"
#include "Symbol/Symbol.cuh"

namespace Test {
    namespace {
        testing::AssertionResult is_integral_solution(const std::string integral_str,
                                                      const std::string expected_result_str) {
            const auto integral = Sym::integral(Parser::parse_function(integral_str));

            Sym::Integrator integrator;
            const auto result = integrator.solve_integral(integral);

            auto expected_result = Parser::parse_function(expected_result_str);

            simplify_vector(expected_result);

            if (!result.has_value()) {
                return testing::AssertionFailure()
                       << "Tried to calculate the integral:\n  " << integral.data()->to_string()
                       << "\n  but no result was found. The result should be:\n  "
                       << expected_result.data()->to_string();
            }

            if (Sym::Symbol::are_expressions_equal(*result.value().data(),
                                                   *expected_result.data())) {
                return testing::AssertionSuccess();
            }

            return testing::AssertionFailure()
                   << "Tried to calculate the integral:\n  " << integral.data()->to_string()
                   << "\n  but got an unexpected result:\n  " << result.value().data()->to_string()
                   << " <- got\n  " << expected_result.data()->to_string() << " <- expected\n";
        }
    }

    void test_integral(const std::string integral_str, const std::string expected_result_str) {
        EXPECT_TRUE(is_integral_solution(integral_str, expected_result_str));
    }
}
