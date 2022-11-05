#include <gtest/gtest.h>

#include "Evaluation/Integrate.cuh"
#include "Evaluation/StaticFunctions.cuh"

#include "Parser/Parser.cuh"

class IntegralCalculation : public ::testing::Test {
  protected:
    IntegralCalculation() { Sym::Static::init_functions(); }
};

void test_integral(const std::string integral_str, const std::string expected_result_str) {
    const auto integral = Sym::integral(parse_function(integral_str));
    const auto result = Sym::solve_integral(integral);
    const auto expected_result = parse_function(expected_result_str);
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(Sym::Symbol::compare_trees(
        result.value().data(), // NOLINT(bugprone-unchecked-optional-access)
        expected_result.data()));
}

TEST_F(IntegralCalculation, X) { test_integral("x", "0.5*x^2"); }
TEST_F(IntegralCalculation, EToX) { test_integral("e^x", "e^x"); }
TEST_F(IntegralCalculation, EToXChain) { test_integral("e^e^x*e^x*e^e^e^x", "e^e^e^x"); }
TEST_F(IntegralCalculation, CosPlusSin) { test_integral("cos(x)+sin(x)", "sin(x)-cos(x)"); }
