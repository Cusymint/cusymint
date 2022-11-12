#include <gtest/gtest.h>

#include "Evaluation/Integrate.cuh"
#include "Parser/Parser.cuh"
#include "Symbol/Symbol.cuh"

#define SIMPLIFY_TEST(_name, _input, _expected) \
    TEST(SimplifyTest, _name) { EXPECT_TRUE(simplifies_to(_input, _expected)); }

#define SIMPLIFY_TEST_NO_ACTION(_name, _input) SIMPLIFY_TEST(_name, _input, _input)

namespace Test {
    namespace {
        testing::AssertionResult
        simplifies_to(const std::vector<Sym::Symbol>& expression,
                      const std::vector<Sym::Symbol>& expected_simplification) {
            auto simplified_expression = expression;
            // Sometimes simplified expressions take more space than before, so this is necessary
            simplified_expression.resize(Sym::EXPRESSION_MAX_SYMBOL_COUNT);

            std::vector<Sym::Symbol> simplification_memory(Sym::EXPRESSION_MAX_SYMBOL_COUNT);
            simplified_expression.data()->simplify(simplification_memory.data());
            simplified_expression.resize(simplified_expression.data()->size());

            if (Sym::Symbol::compare_trees(simplified_expression.data(),
                                           expected_simplification.data())) {
                return testing::AssertionSuccess();
            }

            return testing::AssertionFailure()
                   << "Tried to simplify expression:\n  " << expression.data()->to_string()
                   << "\n  but got an unexpected result:\n  "
                   << simplified_expression // NOLINT(bugprone-unchecked-optional-access)
                          .data()
                          ->to_string()
                   << " <- got\n  " << expected_simplification.data()->to_string()
                   << " <- expected\n";
        }

        testing::AssertionResult simplifies_to(const std::string& expression_str,
                                               const std::string& expected_simplification_str) {
            auto expression = Parser::parse_function(expression_str);
            const auto expected_simplification =
                Parser::parse_function(expected_simplification_str);

            return simplifies_to(expression, expected_simplification);
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
    SIMPLIFY_TEST_NO_ACTION(NoActionReciprocal, "132/x")
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

    SIMPLIFY_TEST(NumberAddition, "10+5+2+3", "20")
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
    SIMPLIFY_TEST(ComplicatedSum, "0+cos(x)^2+10-(e^e^x)+sin(x)^2-0+e^e^x", "11")
    SIMPLIFY_TEST(LongFraction, "(1+3+4)/(4-5+2/(6-1-4/(2-1-1+1)))", "8")
    // Negation distribution inverses terms order 
    SIMPLIFY_TEST(NegationDistribution, "-(e+x+cos(x))", "-cos(x)-x-e")

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
    SIMPLIFY_TEST(OnePower, "(x+e*cos(x))^1", "x+e*cos(x)")
    SIMPLIFY_TEST(PowerOfPower, "((x^10)^pi)^e", "x^(10*pi*e)")

    SIMPLIFY_TEST(SineOfArcsine, "sin(arcsin(10))", "10")
    SIMPLIFY_TEST(CosineOfArccosine, "cos(arccos(x))", "x")
    SIMPLIFY_TEST(TangentOfArctangen, "tan(arctan(e))", "e")
    SIMPLIFY_TEST(CotangentOfArccotangent, "cot(arccot(5^x))", "5^x")
    SIMPLIFY_TEST(PythagoreanTrigIdentity, "cos(x+e^cos(x))^2+sin(x+e^cos(x))^2", "1")
    SIMPLIFY_TEST(AdvancedPythagoreanTrigIdentity, "cos(2)^2+sin(1+cos(x)^2+sin(x)^2)^2", "1")

    SIMPLIFY_TEST(LogarithmOfE, "ln(e)", "1")
    SIMPLIFY_TEST(LogarithmOfOne, "ln(1)", "0")
    SIMPLIFY_TEST(EToLogarithm, "e^ln(x)", "x")
    SIMPLIFY_TEST(PowerInLogarithm, "ln(10^x)", "ln(10)*x")
    SIMPLIFY_TEST(PowerOfLogarithmReciprocal, "10^(1/ln(10))", "e")
    SIMPLIFY_TEST(PowerWithLogarithm, "e^(sin(x)*x*ln(10)*pi)", "10^(sin(x)*x*pi)")
    SIMPLIFY_TEST(PowerWithLogarithmReciprocal, "10^(sin(x)*x/ln(10)*pi)", "e^(sin(x)*x*pi)")
}
