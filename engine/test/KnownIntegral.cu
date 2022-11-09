#include "IntegralCommons.cuh"

namespace Test {
    class KnownIntegral : public IntegrationFixture {};

    TEST_F(KnownIntegral, SimpleVariable) { test_integral("x", "0.5*x^2"); }

    TEST_F(KnownIntegral, SimplePowerFunction) { test_integral("x^4", "(1/5)*x^5"); }

    TEST_F(KnownIntegral, PowerFunctionWithConstant) {
        test_integral("x^pi", "(1/(pi+1))*x^(pi+1)");
    }

    TEST_F(KnownIntegral, AdvancedPowerFunction) {
        test_integral("x^(10+pi*e)", "(1/(pi*e+11))*x^(pi*e+11)");
    }

    TEST_F(KnownIntegral, SimpleExponent) { test_integral("e^x", "e^x"); }

    TEST_F(KnownIntegral, Sine) { test_integral("sin(x)", "-cos(x)"); }

    TEST_F(KnownIntegral, Cosine) { test_integral("cos(x)", "sin(x)"); }

    TEST_F(KnownIntegral, Arctangent) { test_integral("1/(1+x^2)", "arctan(x)"); }

    TEST_F(KnownIntegral, SimpleConstant) { test_integral("10", "x*10"); }

    TEST_F(KnownIntegral, AdvancedConstant) {
        test_integral("pi^e+10*e*ln(pi)", "x*(pi^e+10*e*ln(pi))");
    }

    /* is_constant_integral, is_simple_arctan, */
};
