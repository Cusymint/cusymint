#include "IntegralCommons.cuh"

#define HEURISTIC_TEST(_name, _integral, _expected_result) \
    INTEGRAL_TEST(Heuristic, _name, _integral, _expected_result)

namespace Test {
    class Heuristic : public IntegrationFixture {};

    HEURISTIC_TEST(EToX, "e^x*cos(e^x)", "sin(e^x)")
    HEURISTIC_TEST(EToXTower, "e^x*e^e^x*e^e^e^x*e^e^e^e^x*e^e^e^e^e^x*e^e^e^e^e^e^x",
                   "e^e^e^e^e^e^x")

    HEURISTIC_TEST(IntegralWithConstant, "10/(1+x^2)", "arctan(x)*10")
    HEURISTIC_TEST(Polynomial, "x^3+3*x^2+8*x+12", "(1/4)*(x^4)+x^3+(x^2)*4+x*12")
    HEURISTIC_TEST(ArbitraryProduct, "pi*2*e*e^x*10*sin(e^x)*ln(pi)", "-1*cos(e^x)*ln(pi)*20*e*pi")

    HEURISTIC_TEST(SumIntegral, "cos(x)+sin(x)", "sin(x)-cos(x)")
    HEURISTIC_TEST(LongSumIntegral, "1+cos(x)+sin(x)+1/(1+x^2)+pi+e",
                   "sin(x)-cos(x)+arctan(x)+x*(pi+e+1)")

    HEURISTIC_TEST(SineSubstitution, "cos(x)*e^sin(x)", "e^sin(x)")
    HEURISTIC_TEST(CosineSubstitution, "5cos^4(x)sin(x)", "-1*cos^5(x)")

    // TODO: Universal substitution when simplification is powerful enough
};
