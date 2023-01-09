#include "IntegralCommons.cuh"

#define KNOWN_INTEGRAL_TEST(_name, _integral, _expected_result) \
    INTEGRAL_TEST(KnownIntegral, _name, _integral, _expected_result)

namespace Test {
    class KnownIntegral : public IntegrationFixture {};

    KNOWN_INTEGRAL_TEST(SimpleVariable, "x", "0.5*x^2")

    KNOWN_INTEGRAL_TEST(SimplePowerFunction, "x^4", "(1/5)*x^5")
    KNOWN_INTEGRAL_TEST(PowerFunctionWithConstant, "x^pi", "(1/(pi+1))*x^(pi+1)")
    KNOWN_INTEGRAL_TEST(AdvancedPowerFunction, "x^(10+pi*e)", "(1/(pi*e+11))*x^(pi*e+11)")
    KNOWN_INTEGRAL_TEST(VeryAdvancedPowerFunction, "x^(10*pi*e*sgn(e^(-x^2)))",
                        "(1/(1+10*pi*e*sgn(e^(-x^2))))*x^(1+10*pi*e*sgn(e^(-x^2)))")

    KNOWN_INTEGRAL_TEST(SimpleExponent, "e^x", "e^x")

    KNOWN_INTEGRAL_TEST(Sine, "sin(x)", "-cos(x)")
    KNOWN_INTEGRAL_TEST(Cosine, "cos(x)", "sin(x)")

    KNOWN_INTEGRAL_TEST(Arctangent, "1/(1+x^2)", "arctan(x)")

    KNOWN_INTEGRAL_TEST(SimpleConstant, "10", "x*10")
    KNOWN_INTEGRAL_TEST(AdvancedConstant, "pi^e+10*e*ln(pi)", "x*(pi^e+10*e*ln(pi))")
    KNOWN_INTEGRAL_TEST(AlmostConstant, "100*sgn(e^x^2)", "x*100*sgn(e^x^2)")

    KNOWN_INTEGRAL_TEST(Reciprocal, "1/x", "ln(x)")
    KNOWN_INTEGRAL_TEST(NegativeOnePower, "x^(-1)", "ln(x)")

    KNOWN_INTEGRAL_TEST(Arcsine, "1/sqrt(1-x^2)", "arcsin(x)")

    KNOWN_INTEGRAL_TEST(Tangent, "1/cos^2(x)", "tg(x)")
    KNOWN_INTEGRAL_TEST(Cotangent, "1/sin^2(x)", "-ctg(x)")

    KNOWN_INTEGRAL_TEST(PowerWithSimpleConstantBase, "2^x", "2^x/ln(2)")
    KNOWN_INTEGRAL_TEST(PowerWithAdvancedConstantBase, "(cos(pi)*e^c+1)^x",
                        "(cos(pi)*e^c+1)^x/ln(cos(pi)*e^c+1)")

    KNOWN_INTEGRAL_TEST(ErrorFunction, "e^(-x^2)", "sqrt(pi)/2*erf(x)")
    KNOWN_INTEGRAL_TEST(SineIntegral, "sin(x)/x", "Si(x)")
    KNOWN_INTEGRAL_TEST(CosineIntegral, "cos(x)/x", "Ci(x)")
    KNOWN_INTEGRAL_TEST(LogarithmicIntegral, "1/ln(x)", "li(x)")
    KNOWN_INTEGRAL_TEST(ExponentialIntegral, "e^x/x", "Ei(x)")
};
