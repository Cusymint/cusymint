#include "IntegralCommons.cuh"

#define ADVANCED_INTEGRAL_TEST(_name, _integral, _expected_result) \
    INTEGRAL_TEST(AdvancedIntegral, _name, _integral, _expected_result)

namespace Test {
    class AdvancedIntegral : public IntegrationFixture {};

    // https://pages.mini.pw.edu.pl/~dembinskaa/www/?download=Inf_I_cw9_2022_23.pdf
    ADVANCED_INTEGRAL_TEST(SimpleRationalWithSquare, "(sqrt(x)-2)^2/x^2", "-4/x-6/sqrt(x)+ln(x)")
    ADVANCED_INTEGRAL_TEST(SineCosineSquaredInDenominator, "1/(sin^2(x)cos^2(x))", "-1/tg(x)+tg(x)")

    ADVANCED_INTEGRAL_TEST(LogarithmDividedByX, "ln^5(x)/x", "ln^6(x)/6")
    ADVANCED_INTEGRAL_TEST(ArctangentWithX3Substitution, "x^3/(x^8+1)", "arctg(x^4)/4")
    ADVANCED_INTEGRAL_TEST(LogarithmMultipliedByX, "x^n*ln(x)", "x^(n+1)ln(x)/(n+1)-x^(n+1)/(n+1)^2")
    ADVANCED_INTEGRAL_TEST(ArctangentMultipliedByX, "x*arctg(x)", "x^2*arctg(x)/2-x+arctg(x)")
    ADVANCED_INTEGRAL_TEST(Arcsine, "arcsin(x)", "x*arcsin(x)+2*sqrt(1-x^2)")
}