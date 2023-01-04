#include "IntegralCommons.cuh"

#define ADVANCED_INTEGRAL_TEST(_name, _integral, _expected_result) \
    INTEGRAL_TEST(AdvancedIntegral, _name, _integral, _expected_result)

namespace Test {
    class AdvancedIntegral : public IntegrationFixture {};

    // https://pages.mini.pw.edu.pl/~dembinskaa/www/?download=Inf_I_cw9_2022_23.pdf, tasks 1-3
    ADVANCED_INTEGRAL_TEST(SimpleRationalWithSquare, "(sqrt(x)-2)^2/x^2", "-4/x-6/sqrt(x)+ln(x)")
    ADVANCED_INTEGRAL_TEST(SineCosineSquaredInDenominator, "1/(sin^2(x)cos^2(x))", "-1/tg(x)+tg(x)")

    ADVANCED_INTEGRAL_TEST(LogarithmDividedByX, "ln^5(x)/x", "ln^6(x)/6")
    ADVANCED_INTEGRAL_TEST(ArctangentWithX3Substitution, "x^3/(x^8+1)", "arctg(x^4)/4") // substitution t=x^4 required
    ADVANCED_INTEGRAL_TEST(LogarithmMultipliedByX, "x^n*ln(x)", "x^(n+1)ln(x)/(n+1)-x^(n+1)/(n+1)^2")
    ADVANCED_INTEGRAL_TEST(ArctangentMultipliedByX, "x*arctg(x)", "x^2*arctg(x)/2-x+arctg(x)")
    ADVANCED_INTEGRAL_TEST(Arcsine, "arcsin(x)", "x*arcsin(x)+2*sqrt(1-x^2)") // substitution t=x^2+a required
    ADVANCED_INTEGRAL_TEST(LogarithmWithX2Substitution, "x*ln(x^2+1)", "(x^2+1)ln(x^2+1)-x^2-1")
    ADVANCED_INTEGRAL_TEST(CyclicIntegral1, "e^x*sin(2x)", "e^x*sin(2x)/5 - e^x*2cos(2x)/5") // cyclic integrals required
    ADVANCED_INTEGRAL_TEST(CyclicIntegral2, "sin(ln(x))", "-x/2(cos(ln(x))-sin(ln(x)))")
    ADVANCED_INTEGRAL_TEST(CotangentDividedByLogOfSine, "ctg(x)/ln(sin(x))", "ln(ln(sin(x)))") // substitution t=ln(x) required
    ADVANCED_INTEGRAL_TEST(InvertedCosH, "1/(e^x+e^-x)", "arctg(e^x)")

    ADVANCED_INTEGRAL_TEST(Absolute, "abs(x)", "x*abs(x)/2")
    ADVANCED_INTEGRAL_TEST(MaxOfOneAndSquare, "(abs(1-x^2)+1+x^2)/2", "(sgn(1-x^2)(x-x^3/3)+x+x^3/3)/2")

    // tasks 5-7 involve rational integrals

    // https://pages.mini.pw.edu.pl/~dembinskaa/www/?download=Inf_I_PowtKol3_2022-2023.pdf, task 11
    ADVANCED_INTEGRAL_TEST(TangentSquared, "tg^2(x)", "tg(x)-x")
    ADVANCED_INTEGRAL_TEST(LongPolynomial, "(2x-3)^10", "1/22(2x-3)^11")
    ADVANCED_INTEGRAL_TEST(SquareRootInDenominator, "1/(2+sqrt(x))", "2sqrt(x)-4ln(sqrt(x)+2)")
    ADVANCED_INTEGRAL_TEST(InvertedCosH2, "1/cosh(x)", "2arctg(e^x)")
    ADVANCED_INTEGRAL_TEST(Sine5Cosine, "sin^5(x)cos(x)", "sin^6(x)/6")
}