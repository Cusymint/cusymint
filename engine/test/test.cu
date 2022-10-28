#include <gtest/gtest.h>

#include "Evaluation/Integrate.cuh"
#include "Evaluation/StaticFunctions.cuh"

#include "Parser/Parser.cuh"

class Fixture : public ::testing::Test {
  protected:
    Fixture() { Sym::Static::init_functions(); }
};

TEST(Tests, Test) { EXPECT_EQ(1, 1 + 1 - 1); }

TEST_F(Fixture, CusymintTest) {
    const auto integral = Sym::integral(parse_function("x"));
    EXPECT_EQ("SubexpressionVacancy{ solved by 1 }",
              Sym::solve_integral(integral).value()[0].data()->to_string());
}
