#ifndef INTEGRAL_COMMONS_CUH
#define INTEGRAL_COMMONS_CUH

#include <gtest/gtest.h>

#include <string>

#include "Evaluation/StaticFunctions.cuh"

namespace Test {
    class IntegrationFixture : public ::testing::Test {
      protected:
        IntegrationFixture() { Sym::Static::init_functions(); }
    };

    void test_integral(const std::string integral_str, const std::string expected_result_str);
}

#define INTEGRAL_TEST(_fixture, _name, _integral, _expected_result) \
    TEST_F(_fixture, _name) { test_integral(_integral, _expected_result); }

#endif
