#include <gtest/gtest.h>

#include <vector>

#include "Evaluation/Integrator.cuh"
#include "Parser/Parser.cuh"
#include "Simplify.cuh"
#include "Symbol/Constants.cuh"
#include "Symbol/Symbol.cuh"
#include "Symbol/Variable.cuh"

#define SIMPLIFY_TEST(_name, _input, _expected) \
    TEST(Simplify, _name) { EXPECT_TRUE(simplifies_to(_input, _expected)); }

#define SIMPLIFY_TEST_NO_ACTION(_name, _input) SIMPLIFY_TEST(_name, _input, _input)

#define EQUALITY_TEST(_name, _expression1, _expression2) \
    TEST(Simplify, _name) { EXPECT_TRUE(are_equal(_expression1, _expression2)); }

namespace Test {
    void simplify_vector(std::vector<Sym::Symbol>& expression) {
        while (true) {
            std::vector<Sym::Symbol> expr_copy(expression);
            std::vector<Sym::Symbol> simplification_memory(Sym::Integrator::HELP_SPACE_MULTIPLIER *
                                                           expr_copy.size());

            auto help_space_it = Sym::SymbolIterator::from_at(*simplification_memory.data(), 0,
                                                              simplification_memory.size())
                                     .good();
            const auto result = expr_copy.data()->simplify(help_space_it);

            if (result.is_good()) {
                expr_copy.data()->copy_to(*expression.data());
                break;
            }

            // Sometimes simplified expressions take more space than before, so this is
            // necessary
            expression.resize(Sym::Integrator::REALLOC_MULTIPLIER * expression.size());
        }
        expression.resize(expression.data()->size());
    }

    namespace {
        std::vector<Sym::Symbol> simplify(std::vector<Sym::Symbol> expression) {
            simplify_vector(expression);
            return expression;
        }

        testing::AssertionResult
        simplifies_to(const std::vector<Sym::Symbol>& expression,
                      const std::vector<Sym::Symbol>& expected_simplification) {
            const auto simplified_expression = simplify(expression);

            if (Sym::Symbol::are_expressions_equal(*simplified_expression.data(),
                                                   *expected_simplification.data())) {
                return testing::AssertionSuccess();
            }

            return testing::AssertionFailure()
                   << "Tried to simplify expression:\n  " << expression.data()->to_string()
                   << "\n  but got an unexpected result:\n  "
                   << simplified_expression.data()->to_string() << " <- got\n  "
                   << expected_simplification.data()->to_string() << " <- expected\n";
        }

        testing::AssertionResult simplifies_to(const std::string& expression_str,
                                               const std::string& expected_simplification_str) {
            auto expression = Parser::parse_function(expression_str);
            const auto expected_simplification =
                Parser::parse_function(expected_simplification_str);

            return simplifies_to(expression, expected_simplification);
        }

        testing::AssertionResult
        simplifies_to(const std::string& expression_str,
                      const std::vector<Sym::Symbol>& expected_simplification) {
            auto expression = Parser::parse_function(expression_str);

            return simplifies_to(expression, expected_simplification);
        }

        testing::AssertionResult are_equal(const std::vector<Sym::Symbol>& expression1,
                                           const std::vector<Sym::Symbol>& expression2) {
            auto simplified_expression1 = simplify(expression1);
            auto simplified_expression2 = simplify(expression2);

            if (Sym::Symbol::are_expressions_equal(*simplified_expression1.data(),
                                                   *simplified_expression2.data())) {
                return testing::AssertionSuccess();
            }

            return testing::AssertionFailure()
                   << "Tried to simplify and assert equality of expressions:\n  "
                   << expression1.data()->to_string() << "\n  " << expression2.data()->to_string()
                   << "\n  but they got simplified to:\n  "
                   << simplified_expression1.data()->to_string() << "\n  "
                   << simplified_expression2.data()->to_string() << "\n";
        }

        testing::AssertionResult are_equal(const std::string& expression1_str,
                                           const std::string& expression2_str) {
            const auto expression1 = Parser::parse_function(expression1_str);
            const auto expression2 = Parser::parse_function(expression2_str);

            return are_equal(expression1, expression2);
        }

        testing::AssertionResult are_equal(const std::string& expression1_str,
                                           const std::vector<Sym::Symbol>& expression2) {
            const auto expression1 = Parser::parse_function(expression1_str);

            return are_equal(expression1, expression2);
        }
    }

    SIMPLIFY_TEST_NO_ACTION(NoActionSubexpressionCandidate,
                            Sym::first_expression_candidate(Sym::num(10)));
    SIMPLIFY_TEST_NO_ACTION(NoActionSubexpressionIntegral, Sym::integral(Sym::var()));
    SIMPLIFY_TEST_NO_ACTION(NoActionSubexpressionSolution, Sym::solution(Sym::var()));
    SIMPLIFY_TEST_NO_ACTION(NoActionSubexpressionVacancy, Sym::single_integral_vacancy())
    SIMPLIFY_TEST_NO_ACTION(NoActionX, "x")
    SIMPLIFY_TEST_NO_ACTION(NoActionNum, "10")
    SIMPLIFY_TEST_NO_ACTION(NoActionKnownConstant, "pi")
    SIMPLIFY_TEST_NO_ACTION(NoActionExp, "e^x")
    SIMPLIFY_TEST_NO_ACTION(NoActionConst, "a")
    SIMPLIFY_TEST_NO_ACTION(NoActionReciprocal, Sym::num(132) * Sym::inv(Sym::var()))
    SIMPLIFY_TEST_NO_ACTION(NoActionSine, "sin(cos(x))")
    SIMPLIFY_TEST_NO_ACTION(NoActionCosine, "cos(e)")
    SIMPLIFY_TEST_NO_ACTION(NoActionTangen, "tan(x)")
    SIMPLIFY_TEST_NO_ACTION(NoActionCotangen, "cot(71830)")
    SIMPLIFY_TEST_NO_ACTION(NoActionArcsine, "arcsin(x^2)")
    SIMPLIFY_TEST_NO_ACTION(NoActionArccos, "arccos(c)")
    SIMPLIFY_TEST_NO_ACTION(NoActionArctan, "arctan(x)")
    SIMPLIFY_TEST_NO_ACTION(NoActionArccot, "arccot(10)")
    SIMPLIFY_TEST_NO_ACTION(NoActionLn, "ln(5)")
    SIMPLIFY_TEST_NO_ACTION(NoActionLong, "ln(10)^x^10^e^cos(e^pi)^sin(cos(tan(cot(x))))")

    SIMPLIFY_TEST(NumberAddition, "10+1+2+3+20+5+9+11+1", "62")
    // Parsing creates `Negation` symbol, so explicit symbol creation necessary
    SIMPLIFY_TEST(NumberNegation, -Sym::num(1), Sym::num(-1))
    SIMPLIFY_TEST(NumberSubtraction, Sym::num(10) - Sym::num(30), Sym::num(-20))
    SIMPLIFY_TEST(NumberMultiplication, "10*4*3", "120")
    SIMPLIFY_TEST(NumberPower, "2^10", "1024")

    SIMPLIFY_TEST(ZeroAdditionLeft, "0+x", "x")
    SIMPLIFY_TEST(ZeroAdditionRIght, "x+0", "x")
    SIMPLIFY_TEST(MultipleEvenNegations, "----x", "x")
    SIMPLIFY_TEST(MultipleOddNegations, "-----x", "-x")
    SIMPLIFY_TEST(IdenticalSubtraction, "(sin(x)+cos(x)-20^e)-(sin(x)+cos(x)-20^e)", "0")
    SIMPLIFY_TEST(OneReciprocal, "tan(x)/1", "tan(x)")
    EQUALITY_TEST(ComplicatedSum, "cos(x)^2+10-x+sin(x)^2+x", "11")
    SIMPLIFY_TEST(LongFraction, "(1+3+4)/(4-5+2/(6-1-4/(2-1-1+1)))", "8")
    // Negation distribution inverses terms order
    EQUALITY_TEST(NegationDistribution, "-(e+x+cos(x))", "-cos(x)-x-e")
    EQUALITY_TEST(NegationDistributionWithNegation, "-(e-pi^x+x+sin(x)-tan(x))",
                  "tan(x)-sin(x)-x+pi^x-e")

    SIMPLIFY_TEST(OneMultiplicationLeft, "1*x", "x")
    SIMPLIFY_TEST(OneMultiplicationRight, "x*1", "x")
    SIMPLIFY_TEST(ZeroMultiplicationLeft, "0*x", "0")
    SIMPLIFY_TEST(ZeroMultiplicationRight, "x*0", "0")
    SIMPLIFY_TEST(LongZeroMultiplication, "x*0*e^cos(sin(x))*a*pi*b*c*(x^x+7)", "0")
    SIMPLIFY_TEST(MultipleEvenReciprocals, "1/(1/(1/(1/(x))))", "x")
    SIMPLIFY_TEST(MultipleOddReciprocals, Sym::inv(Sym::inv(Sym::inv(Sym::var()))),
                  Sym::inv(Sym::var()))
    SIMPLIFY_TEST(IdenticalQuotient, "(sin(x)+cos(x)-20^e^x+pi)/(sin(x)+cos(x)-20^e^x+pi)", "1")
    SIMPLIFY_TEST(ComplicatedProduct, "1*(1/x)*2*1*(1/10)*x*10/1", "2")

    SIMPLIFY_TEST(ZeroPower, "(10*x+pi-cos(sin(x)))^0", "1")
    EQUALITY_TEST(OnePower, "(x+e*cos(x))^1", "x+e*cos(x)")
    SIMPLIFY_TEST(PowerOfPower, "((x^10)^pi)^e", "x^(10*pi*e)")

    SIMPLIFY_TEST(SineOfArcsine, "sin(arcsin(10))", "10")
    SIMPLIFY_TEST(CosineOfArccosine, "cos(arccos(x))", "x")
    SIMPLIFY_TEST(TangentOfArctangen, "tan(arctan(e))", "e")
    SIMPLIFY_TEST(CotangentOfArccotangent, "cot(arccot(5^x))", "5^x")
    SIMPLIFY_TEST(PythagoreanTrigIdentity, "cos(x+e^cos(x))^2+sin(x+e^cos(x))^2", "1")
    SIMPLIFY_TEST(AdvancedPythagoreanTrigIdentity, "cos(2)^2+sin(1+cos(x)^2+sin(x)^2)^2", "1")
    SIMPLIFY_TEST(TangentCotangentProduct, "tg(tan(x)cot(x)x^2)ctg(x^2)", "1")

    SIMPLIFY_TEST(LogarithmOfE, "ln(e)", "1")
    SIMPLIFY_TEST(LogarithmOfOne, "ln(1)", "0")
    SIMPLIFY_TEST(EToLogarithm, "e^ln(x)", "x")
    SIMPLIFY_TEST(PowerInLogarithm, "ln(10^x)", "x*ln(10)")
    SIMPLIFY_TEST(PowerOfLogarithmReciprocal, "10^(1/ln(10))", "e")
    EQUALITY_TEST(PowerWithLogarithm, "e^(sin(x)*x*ln(10)*pi)", "10^(sin(x)*x*pi)")
    EQUALITY_TEST(PowerWithLogarithmReciprocal, "10^(sin(x)*x/ln(10)*pi)", "e^(sin(x)*x*pi)")

    SIMPLIFY_TEST(PolynomialsOfEqualRank, "(9+2*x^2+x^3)/(3+x+5*x^2+10*x^3)",
                  Sym::num(0.1) +
                      Sym::num(-0.1) * Sym::var() / Parser::parse_function("(3+x+5*x^2+10*x^3)") +
                      Parser::parse_function("1.5x^2/(3+x+5*x^2+10*x^3)") +
                      Parser::parse_function("8.7/(3+x+5*x^2+10*x^3)"))
    SIMPLIFY_TEST(
        NoActionNumeratorRankLessThanDenominator, "(9+2*x^2+x^3)/(3+x+5*x^2+10*x^3+x^6)",
        "(2*x^2)/(3+x+5*x^2+10*x^3+x^6)+(x^3)/(3+x+5*x^2+10*x^3+x^6)+(9)/(3+x+5*x^2+10*x^3+x^6)")
    SIMPLIFY_TEST(DivisiblePolynomials, "(x^4-1)/(x^2+1)",
                  Sym::num(-1) + (Sym::var() ^ Sym::num(2)))
    SIMPLIFY_TEST(DivideMonomialByMonomial, "x^5/x", "x^4")
    SIMPLIFY_TEST(DivideByConstant, "x^5/2", "0.5*x^5")
    SIMPLIFY_TEST_NO_ACTION(DivideConstantByPolynomial, "2/(1-x^2+x^5)")
    SIMPLIFY_TEST(DivideAdvancedMonomialByMonomial, "2*5*7*x^5/(---(14*x^2))",
                  Sym::num(-5) * (Sym::var() ^ Sym::num(3)))
    SIMPLIFY_TEST(PolynomialsDivisibleWithRemainder, "x^4/(x^2+1)",
                  Sym::num(-1) + Sym::inv(Sym::num(1) + (Sym::var() ^ Sym::num(2))) +
                      (Sym::var() ^ Sym::num(2)))
    SIMPLIFY_TEST(LongPolynomialsDivisibleWithRemainder, "(x^5+6*x^2+x+9)/(x^2+x+1)",
                  Sym::num(7) + (Sym::num(-6) * Sym::var()) / Parser::parse_function("1+x+x^2") +
                      Parser::parse_function("2/(1+x+x^2)") +
                      (Sym::num(-1) * (Sym::var() ^ Sym::num(2))) + (Sym::var() ^ Sym::num(3)))

    EQUALITY_TEST(SameExpressionsAddition, "x+x", "2*x")
    EQUALITY_TEST(SameExpressionsAdditionBeingSorted, "sin(x)+ln(x)+sin(x)", "2*sin(x)+ln(x)")
    EQUALITY_TEST(SameExpressionsAdditionBeingSorted2, "ln(x)+sin(x)+ln(x)", "sin(x)+2*ln(x)")
    EQUALITY_TEST(SameExpressionsAdditionLeftMultiplied, "3*c+c", "4*c")
    EQUALITY_TEST(SameExpressionsAdditionRightMultiplied, "e^x+8*e^x", "9*e^x")
    EQUALITY_TEST(SameExpressionsMultipliedByConstantAddition, "5*cos(x)+10*cos(x)", "15*cos(x)")
    EQUALITY_TEST(ReducingSameExpressionsAddition, "x*y-x*y", "0")

    EQUALITY_TEST(SameLongExpressionsAddition, "x*y*z+x*y*z", "2*x*y*z")
    EQUALITY_TEST(SameLongExpressionsAdditionBeingSorted, "sin(x)*cos(x)*x+ln(x)+x*cos(x)*sin(x)",
                  "2*sin(x)*cos(x)*x+ln(x)")
    EQUALITY_TEST(SameLongExpressionsAdditionBeingSorted2, "tan(x)*v*ln(x)+sin(x)+ln(x)*tan(x)*v",
                  "sin(x)+2*ln(x)*tan(x)*v")
    EQUALITY_TEST(SameLongExpressionsAdditionLeftMultiplied, "3*c*a*b*d*h+c*a*b*d*h", "4*c*a*b*d*h")
    EQUALITY_TEST(SameLongExpressionsAdditionRightMultiplied, "e^x*a*b*h*d+8*e^x*b*a*d*h",
                  "9*e^x*a*b*d*h")
    EQUALITY_TEST(SameLongExpressionsMultipliedByConstantAddition, "5*cos(x)*a*b*c+10*c*cos(x)*b*a",
                  "15*a*cos(x)*c*b")
    EQUALITY_TEST(ReducingSameLongExpressionsAddition, "x*y*b*n*e*u*l*j-x*y*n*b*l*j*e*u", "0")

    EQUALITY_TEST(SameExpressionsMultiplication, "x*x", "x^2")
    EQUALITY_TEST(SameExpressionsMultiplicationBeingSorted, "sin(x)*ln(x)*sin(x)", "sin(x)^2*ln(x)")
    EQUALITY_TEST(SameExpressionsMultiplicationBeingSorted2, "ln(x)*sin(x)*ln(x)", "sin(x)*ln(x)^2")
    EQUALITY_TEST(SameExpressionsMultiplicationLeftPowered, "c^3*c", "c^4")
    EQUALITY_TEST(SameExpressionsMultiplicationRightPowered, "e^x*(e^x)^8", "e^(9*x)")
    EQUALITY_TEST(SameExpressionsPoweredToConstantMultiplication, "cos(x)^5*cos(x)^10", "cos(x)^15")
    EQUALITY_TEST(ReducingSameExpressionsMultiplication, "((x*v)^sin(x))/((x*v)^sin(x))", "1")

    EQUALITY_TEST(SameLongExpressionsMultiplication, "x^(b*c*d)*x^(b*c*d)", "x^(2*b*c*d)")
    EQUALITY_TEST(SameLongExpressionsMultiplicationBeingSorted,
                  "sin(x)^(a*b*c)*ln(x)*sin(x)^(3*b*c*a)", "sin(x)^(4*a*b*c)*ln(x)")
    EQUALITY_TEST(SameLongExpressionsMultiplicationBeingSorted2,
                  "ln(x)^(x*x*x)*sin(x)*ln(x)^(x*x*x)", "sin(x)*ln(x)^(2*x*x*x)")
    EQUALITY_TEST(SameLongExpressionsMultiplicationLeftPowered, "c^(3*d)*c^d", "c^(4*d)")
    EQUALITY_TEST(SameLongExpressionsMultiplicationRightPowered, "e^(20*x)*(e^x)^8", "e^(28*x)")
    EQUALITY_TEST(SameLongExpressionsPoweredToConstantMultiplication,
                  "cos(x)^(5*a*b*c*d*e*f)*cos(x)^(10*a*b*c*d*e*f)", "cos(x)^(15*a*b*c*d*e*f)")
    EQUALITY_TEST(ReducingSameLongExpressionsMultiplication,
                  "((x*v)^(18*sin(x)*b))/((x*v)^(b*sin(x)*18))", "1")

    EQUALITY_TEST(ReducingIdenticalFractionWithCoefficients, "(x+1)/(-x-1)", "-1")
    EQUALITY_TEST(ReducingIdenticalFractionWithCoefficients2, "(x+1)/(-2x-2)", "-0.5")
    EQUALITY_TEST(ReducingIdenticalFractionWithCoefficients3, "6(x+1)/(2x+2)", "3")

    EQUALITY_TEST(PoweredProduct, "(sin(x+2)*ln(cos(x)))^(e^x)", "sin(2+x)^e^x*ln(cos(x))^e^x")
    EQUALITY_TEST(PoweredToSum, "(x+1)^(x+c)", "(1+x)^c * (1+x)^x")
    EQUALITY_TEST(PoweredLongProduct, "(x*cos(x)*sin(x)*ln(x)*d)^2",
                  "d^2*x^2*(sin(x))^2*(cos(x))^2*ln(x)^2")
    SIMPLIFY_TEST(LinearPolynomialsDivision, "(2x-1)/x", Sym::num(2) + Sym::num(-1) / Sym::var())
    EQUALITY_TEST(PoweredToLongSum, "e^(1+c+x+cos(x))", "e^x*e^(1+c)*e^cos(x)")

    EQUALITY_TEST(LogarithmOfProduct, "ln(x*ln(x))", "ln(x)+ln(ln(x))")
    EQUALITY_TEST(SplitProductIntoSum, "(x+1)*(y+c)", "c+y+c*x+y*x")

    EQUALITY_TEST(ExpandBinomial, "(x+1)^3", "1+3*x+3*x^2+x^3")
    EQUALITY_TEST(
        ExpandAdvancedExpression, "(tg(x)+x*y+cos(x)+3)^4",
        "((((((((((((((((((((((((((((((((((81+((108*y)*x))+((54*y^2)*x^2))+(((36*y)*x)*cos(x)^2))+("
        "((6*y^2)*x^2)*cos(x)^2))+(((36*y)*x)*tan(x)^2))+(((6*y^2)*x^2)*tan(x)^2))+((6*cos(x)^2)*"
        "tan(x)^2))+((36*cos(x))*tan(x)^2))+((((12*y)*x)*cos(x))*tan(x)^2))+((12*y^3)*x^3))+(((4*y)"
        "*x)*cos(x)^3))+(((4*y)*x)*tan(x)^3))+((4*cos(x))*tan(x)^3))+(y^4*x^4))+(((108*y)*x)*cos(x)"
        "))+(((36*y^2)*x^2)*cos(x)))+(((4*y^3)*x^3)*cos(x)))+(((108*y)*x)*tan(x)))+(((36*y^2)*x^2)*"
        "tan(x)))+((36*cos(x)^2)*tan(x)))+((((12*y)*x)*cos(x)^2)*tan(x)))+(((4*y^3)*x^3)*tan(x)))+("
        "(4*cos(x)^3)*tan(x)))+((108*cos(x))*tan(x)))+((((72*y)*x)*cos(x))*tan(x)))+((((12*y^2)*x^"
        "2)*cos(x))*tan(x)))+(54*cos(x)^2))+(54*tan(x)^2))+(12*cos(x)^3))+(12*tan(x)^3))+cos(x)^4)+"
        "tan(x)^4)+(108*cos(x)))+(108*tan(x)))")

    SIMPLIFY_TEST(SplitReciprocal, "1/(e*x*sin^2(x))",
                  Sym::inv(Sym::e()) / Sym::var() * (Sym::sin(Sym::var()) ^ Sym::num(-2)))

    EQUALITY_TEST(ExponentialIntegralOfLogarithm, "1+Ei(ln(x+2))", "1+li(2+x)")
    EQUALITY_TEST(LogarithmicIntegralOfExponential, "5*li(e^(pi*x))", "5*Ei(pi*x)")

    EQUALITY_TEST(SignOfConstant1, "sgn(-6)", "-1")
    EQUALITY_TEST(SignOfConstant2, "sgn(348)", "1")
    EQUALITY_TEST(SignOfConstant3, "sgn(0)", "0")
    SIMPLIFY_TEST_NO_ACTION(SignOfVariable, "sgn(x)")
    EQUALITY_TEST(OneOverSign, "1/sgn(sin(x))", "sgn(sin(x))")
    EQUALITY_TEST(SquareOfAbsoluteValue, "abs(x)^2", "x^2")
    EQUALITY_TEST(AbsoluteValueOfSquare, "abs(x^2)", "x^2")
    EQUALITY_TEST(EvenPowerOfAbsoluteValue, "abs(x)^78232", "x^78232")
    EQUALITY_TEST(AbsoluteValueOfEvenPower, "abs(x^78232)", "x^78232")
    EQUALITY_TEST(SignOfSign, "sgn(sgn(tan(x)))", "sgn(tan(x))")
    EQUALITY_TEST(AbsoluteValueOfAbsoluteValue, "abs(abs(cot(x)))", "abs(cot(x))")
    EQUALITY_TEST(AbsoluteValueOfNegation, "abs(-x)", "abs(x)")
}
