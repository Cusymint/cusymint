#include "IntegralCommons.cuh"

#define KNOWN_INTEGRAL_TEST(_name, _integral, _expected_result) \
    INTEGRAL_TEST(KnownIntegral, _name, _integral, _expected_result)

namespace Test {
    class KnownIntegral : public IntegrationFixture {};

    KNOWN_INTEGRAL_TEST(SimpleVariable, "x", "0.5*x^2")

    KNOWN_INTEGRAL_TEST(SimplePowerFunction, "x^4", "(1/5)*x^5")
    KNOWN_INTEGRAL_TEST(PowerFunctionWithConstant, "x^pi", "(1/(pi+1))*x^(pi+1)")
    KNOWN_INTEGRAL_TEST(AdvancedPowerFunction, "x^(10+pi*e)", "(1/(pi*e+11))*x^(pi*e+11)")

    KNOWN_INTEGRAL_TEST(SimpleExponent, "e^x", "e^x")

    KNOWN_INTEGRAL_TEST(Sine, "sin(x)", "-cos(x)")
    KNOWN_INTEGRAL_TEST(Cosine, "cos(x)", "sin(x)")

    KNOWN_INTEGRAL_TEST(Arctangent, "1/(1+x^2)", "arctan(x)")

    KNOWN_INTEGRAL_TEST(SimpleConstant, "10", "x*10")
    KNOWN_INTEGRAL_TEST(AdvancedConstant, "pi^e+10*e*ln(pi)", "x*(pi^e+10*e*ln(pi))")
};
