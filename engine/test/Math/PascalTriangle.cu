#include <gtest/gtest.h>

#include "Utils/PascalTriangle.cuh"

#define BINOM_TEST(_name, _n, _i, _val) \
    TEST(PascalTriangleTest, _name) { test_binom_value(_n, _i, _val); }

namespace Test {
    namespace {
        void test_binom_value(size_t n, size_t i, size_t expected_value) {
            size_t* data = new size_t[(n + 1) * (n + 2) / 2];
            auto triangle = Util::PascalTriangle::generate(n, data);

            EXPECT_EQ(triangle.binom(n,i), expected_value);
        }
    }

    BINOM_TEST(ZeroOverZero, 0, 0, 1)
    BINOM_TEST(NumberOverZero, 50, 0, 1)
    BINOM_TEST(NumberOverSameNumber, 32, 32, 1)

    BINOM_TEST(NumberOverOne, 94, 1, 94)
    BINOM_TEST(NumberOverNumberMinusOne, 71, 70, 71)

    BINOM_TEST(ArbitraryNumbers, 43, 21, 1052049481860)
}