#include "IntegralCommons.cuh"

#define ADVANCED_INTEGRAL_TEST(_name, _integral, _expected_result) \
    INTEGRAL_TEST(AdvancedIntegral, _name, _integral, _expected_result)

namespace Test {
    class AdvancedIntegral : public IntegrationFixture {};

    // easy excercises from http://www.math.uni.wroc.pl/~ikrol/zbiorek7.pdf
    ADVANCED_INTEGRAL_TEST(EasyPolynomial, "x^4-5x+3", "x^5/5-5x^2/2+3x")
    ADVANCED_INTEGRAL_TEST(EasyPolynomial2, "(1-x^3)^2", "x-x^4/2+x^7/7")
    ADVANCED_INTEGRAL_TEST(KnownConstantMonomial, "x^3t^2", "x^4t^2/4")
    ADVANCED_INTEGRAL_TEST(Root, "x^(2/3)", "3/5x^(5/3)")
    ADVANCED_INTEGRAL_TEST(RootInverse, "1/x^(2/3)", "3x^(1/3)")
    ADVANCED_INTEGRAL_TEST(SimpleRational, "(x^3-3x^2+1)/x^5", "-1/x+3/2/x^2-1/4/x^4")
    ADVANCED_INTEGRAL_TEST(RootRational, "(x^(2/3)-sqrt(x))/x^(1/3)", "3/4x^(4/3)-6/7x^(7/6)")
    ADVANCED_INTEGRAL_TEST(ExponentialsSum, "2e^x+3*5^x", "2e^x+3*5^x/ln(5)")

    ADVANCED_INTEGRAL_TEST(CosineByParts, "x cos(x)", "x sin(x)+cos(x)")
    ADVANCED_INTEGRAL_TEST(ExponentByParts, "x e^x", "x e^x-e^x")
    ADVANCED_INTEGRAL_TEST(ExponentByParts2, "x^2e^x", "e^x(x^2-2x+2)")
    //ADVANCED_INTEGRAL_TEST(NegativeExponentByParts, "x e^(-x)", "-e^(-x)(x+1)") // linear substitution for nested-vars-only required
    ADVANCED_INTEGRAL_TEST(CosineByParts2, "x^2cos(x)", "x^2sin(x)+2x cos(x)-2sin(x)")
    ADVANCED_INTEGRAL_TEST(LogarithmTimesRoot, "sqrt(x)ln(x)", "2/3x^(3/2)(ln(x)-2/3)")
    ADVANCED_INTEGRAL_TEST(LogarithmDividedByPower, "ln(x)/x^4", "-1/(3x^3)(ln(x)+1/3)")

    ADVANCED_INTEGRAL_TEST(VeryLongPolynomial, "(2x-1)^20", "1/42(2x-1)^21+1/42")
    ADVANCED_INTEGRAL_TEST(CosineOfLinearFunction, "cos(3x-1)", "sin(3x-1)/3")
    //ADVANCED_INTEGRAL_TEST(RationalWithSubstitution, "2x/(5x^2+1)", "1/5ln(5x^2+1)") // substitution t=x^2+a required
    ADVANCED_INTEGRAL_TEST(SimpleRationalWithExponent, "e^x/(3+e^x)", "ln(3+e^x)")
    //ADVANCED_INTEGRAL_TEST(ExponentWithSquare, "x*e^(-x^2)", "-1/2e^(-x^2)") // substitution t=x^2+a required
    //ADVANCED_INTEGRAL_TEST(ExponentOfInverse, "e^(1/x)/x^2", "-e^(1/x)") // substitution t=ax^b+c required (dx = 1/b((t-c)/a)^((1-b)/b))
    ADVANCED_INTEGRAL_TEST(Tangent, "tg(x)", "-ln(cos(x))")
    ADVANCED_INTEGRAL_TEST(TangentDividedByCosSquared, "tg(x)/cos^2(x)", "tg^2(x)/2")
    ADVANCED_INTEGRAL_TEST(CosTimesExpSine, "cos(x)e^sin(x)", "e^sin(x)")
    ADVANCED_INTEGRAL_TEST(CosineDividedByRootOfLinSine, "cos(x)/sqrt(1+sin(x))", "2sqrt(1+sin(x))")
    //ADVANCED_INTEGRAL_TEST(TangentDerivativeWithSubstitution, "x^3/cos^2(x^4)", "tg(x^4)/4") // substitution t=ax^b+c required (dx = 1/b((t-c)/a)^((1-b)/b))
    ADVANCED_INTEGRAL_TEST(ExponentOfLinear, "6^(1-x)", "-6^(1-x)/ln(6)")
    //ADVANCED_INTEGRAL_TEST(ArctanSubtitution, "1/(1+x^2)/arctan(x)", "ln(arctan(x))") // substitution t=arctan(x) required

    // task 4 contains examples which require nested integrals (not implemeneted)

    // harder ones from WUT MiNI faculty
    // https://pages.mini.pw.edu.pl/~dembinskaa/www/?download=Inf_I_cw9_2022_23.pdf, tasks 1-3
    ADVANCED_INTEGRAL_TEST(SimpleRationalWithSquare, "(sqrt(x)-2)^2/x^2", "-4/x+8/sqrt(x)+ln(x)")
    ADVANCED_INTEGRAL_TEST(SineCosineSquaredInDenominator, "1/(sin^2(x)cos^2(x))", "-1/tg(x)+tg(x)")

    //ADVANCED_INTEGRAL_TEST(LogarithmDividedByX, "ln^5(x)/x", "ln^6(x)/6") // substitution t=ln(x) or cyclic integrals required
    //ADVANCED_INTEGRAL_TEST(ArctangentWithX3Substitution, "x^3/(x^8+1)", "arctg(x^4)/4") // substitution t=x^4 required
    //ADVANCED_INTEGRAL_TEST(LogarithmMultipliedByX, "x^n*ln(x)", "x^(n+1)ln(x)/(n+1)-x^(n+1)/(n+1)^2") // non-numeric powers integration by parts required
    ADVANCED_INTEGRAL_TEST(ArctangentMultipliedByX, "x*arctg(x)", "x^2/2*arctg(x)-x/2+arctg(x)/2")
    //ADVANCED_INTEGRAL_TEST(Arcsine, "arcsin(x)", "x*arcsin(x)+2*sqrt(1-x^2)") // substitution t=x^2+a required
    //ADVANCED_INTEGRAL_TEST(LogarithmWithX2Substitution, "x*ln(x^2+1)", "(x^2+1)ln(x^2+1)-x^2-1") // substitution t=x^2+a required
    //ADVANCED_INTEGRAL_TEST(CyclicIntegral1, "e^x*sin(2x)", "e^x*sin(2x)/5 - e^x*2cos(2x)/5") // cyclic integrals required
    //ADVANCED_INTEGRAL_TEST(CyclicIntegral2, "sin(ln(x))", "-x/2(cos(ln(x))-sin(ln(x)))") // cyclic integrals required
    //ADVANCED_INTEGRAL_TEST(CotangentDividedByLogOfSine, "ctg(x)/ln(sin(x))", "ln(ln(sin(x)))") // substitution t=ln(x) required
    //ADVANCED_INTEGRAL_TEST(InvertedCosH, "1/(e^x+e^-x)", "arctg(e^x)") // extract_function required

    ADVANCED_INTEGRAL_TEST(Absolute, "abs(x)", "x*abs(x)/2") // abs required
    ADVANCED_INTEGRAL_TEST(MaxOfOneAndSquare, "(abs(1-x^2)+1+x^2)/2", "(sgn(1-x^2)(x-x^3/3)+x+x^3/3)/2") // abs required

    // tasks 5-7 involve rational integrals

    // https://pages.mini.pw.edu.pl/~dembinskaa/www/?download=Inf_I_PowtKol3_2022-2023.pdf, task 11
    ADVANCED_INTEGRAL_TEST(TangentSquared, "tg^2(x)", "tg(x)-arctg(tg(x))") // may sometimes fail
    ADVANCED_INTEGRAL_TEST(LongPolynomial, "(2x-3)^10", "1/22(2x-3)^11+3^11/22")
    //ADVANCED_INTEGRAL_TEST(SquareRootInDenominator, "1/(2+sqrt(x))", "2sqrt(x)-4ln(sqrt(x)+2)") // substitution t=sqrt(x)+a required
    //ADVANCED_INTEGRAL_TEST(InvertedCosH2, "1/cosh(x)", "2arctg(e^x)") // extract_funcion required
    ADVANCED_INTEGRAL_TEST(Sine5Cosine, "sin^5(x)cos(x)", "sin^6(x)/6")
    //ADVANCED_INTEGRAL_TEST(ExpressionWithSquareInDenominator, "x/(x^2-1)^(3/2)", "-1/sqrt(x^2-1)") // substitution t=ax^b+c required
    //ADVANCED_INTEGRAL_TEST(ArcsineWithDerivative, "arcsin^2(x)/sqrt(1-x^2)", "arcsin^3(x)/3") // substitution t=arcsin(x) required
    //ADVANCED_INTEGRAL_TEST(RootInDenominator, "1/(x^(1/3)+1)", "3/2*x^(2/3)-3x^(1/3)+3ln(x^(1/3)+1)") // substitution t=ax^b+c required (dx = 1/b((t-c)/a)^((1-b)/b))
    ADVANCED_INTEGRAL_TEST(Logarithm, "ln(x)", "x ln(x) - x")
    ADVANCED_INTEGRAL_TEST(XTimesCosine, "x*cos(x)", "x*sin(x)+cos(x)")
    //ADVANCED_INTEGRAL_TEST(XTimesExponential, "x^2e^(1-x)", "-(x^2+2x+2)e^(1-x)") // linear substitution for nested-vars-only required
    //ADVANCED_INTEGRAL_TEST(SquareTimesExponential, "8x^2e^(4-x^3)", "-8/3e^(4-x^3)") // substitution t=x^2+a required

    //ADVANCED_INTEGRAL_TEST(SineSquared, "sin^2(x)", "x/2-sin(2x)/4") // rational integrals required
    ADVANCED_INTEGRAL_TEST(Sine5, "sin^5(x)", "-cos^5(x)/5+2/3cos^3(x)-cos(x)")
    ADVANCED_INTEGRAL_TEST(Sine4Cos3, "sin^4(x)cos^3(x)", "sin^5(x)/5-sin^7(x)/7")
    //ADVANCED_INTEGRAL_TEST(Sine4Cos2, "sin^4(x)cos^2(x)", "1/6sin^5(x)cos(x)+1/24sin^3(x)cos(x)-1/32sin(x)cos(x)+1/16x") // rational integrals required
    //ADVANCED_INTEGRAL_TEST(Cos3XCos5X, "cos(3x)cos(5x)", "1/16sin(8x)+1/4sin(2x)") // more trigonometric identitied required
}