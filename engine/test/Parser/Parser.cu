#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "Parser/Parser.cuh"

#include "Symbol/Constants.cuh"
#include "Symbol/Hyperbolic.cuh"
#include "Symbol/Integral.cuh"
#include "Symbol/InverseTrigonometric.cuh"
#include "Symbol/Logarithm.cuh"
#include "Symbol/Symbol.cuh"
#include "Symbol/Trigonometric.cuh"
#include "Symbol/Variable.cuh"

#define PARSER_TEST(_name, _input, _expected_result) \
    TEST(ParserTest, _name) { test_parse(_input, _expected_result); } // NOLINT

#define PARSER_TEST_ERR(_name, _input) \
    TEST(ParserTest, _name) { test_parser_error(_input); } // NOLINT

namespace {
    void test_parse(std::string input, std::vector<Sym::Symbol> expected_result) {
        auto result = Parser::parse_function(input);
        EXPECT_TRUE(Sym::Symbol::are_expressions_equal(*result.data(), *expected_result.data()))
            << "Invalid parse result:\n\twas: " << result.data()->to_string()
            << ", expression size = " << result.data()->size()
            << ",\n\texpected: " << expected_result.data()->to_string()
            << ", expression size = " << expected_result.data()->size();
    }

    void test_parser_error(std::string input) {
        EXPECT_THROW(Parser::parse_function(input),
                     std::invalid_argument); // NOLINT(hicpp-avoid-goto)
    }
}

namespace Test {
    PARSER_TEST(LessThanOneDoubleWithoutLeadingZero, ".123", Sym::num(0.123))
    PARSER_TEST(LessThanOneDoubleWithLeadingZero, "0.123", Sym::num(0.123))
    PARSER_TEST(GreaterThanOneDoubleWithoutZeroAfterComma, "420.", Sym::num(420))
    PARSER_TEST(GreaterThanOneDouble, "123.123", Sym::num(123.123))
    PARSER_TEST(Integer, "2137", Sym::num(2137))

    PARSER_TEST(SymbolicConstant, "a", Sym::cnst("a"))

    PARSER_TEST(NeperConstant, "e", Sym::e())
    PARSER_TEST(PiConstant, "pi", Sym::pi())

    PARSER_TEST(Variable, "x", Sym::var())

    PARSER_TEST(Negation, "-4", -Sym::num(4))
    PARSER_TEST(DoubleNegation, "--4", -(-Sym::num(4)))

    PARSER_TEST(HyperbolicSine, "sinh(x)", Sym::sinh(Sym::var()))
    PARSER_TEST(HyperbolicCosine, "cosh(x)", Sym::cosh(Sym::var()))
    PARSER_TEST(HyperbolicTangent, "tanh(x)", Sym::tanh(Sym::var()))
    PARSER_TEST(HyperbolicTg, "tgh(x)", Sym::tanh(Sym::var()))
    PARSER_TEST(HyperbolicCotangent, "coth(x)", Sym::coth(Sym::var()))
    PARSER_TEST(HyperbolicCtg, "ctgh(x)", Sym::coth(Sym::var()))

    PARSER_TEST(Arcsine, "arcsin(x)", Sym::arcsin(Sym::var()))
    PARSER_TEST(Arccosine, "arccos(x)", Sym::arccos(Sym::var()))
    PARSER_TEST(Arctangent, "arctan(x)", Sym::arctan(Sym::var()))
    PARSER_TEST(Arctg, "arctg(x)", Sym::arctan(Sym::var()))
    PARSER_TEST(Arccotangent, "arccot(x)", Sym::arccot(Sym::var()))
    PARSER_TEST(Arcctg, "arcctg(x)", Sym::arccot(Sym::var()))

    PARSER_TEST(NaturalLogarithm, "ln(x)", Sym::ln(Sym::var()))
    PARSER_TEST(LogarithmWithArbitraryBase, "log_2(x)", Sym::log(Sym::num(2), Sym::var()))
    PARSER_TEST(LogarithmWithArbitraryBase2, "log_sin(pi)(x)",
                Sym::log(Sym::sin(Sym::pi()), Sym::var()))

    PARSER_TEST(SimpleAddition, "x+2", Sym::var() + Sym::num(2))
    PARSER_TEST(SimpleSubtraction, "x-2", Sym::var() - Sym::num(2))
    PARSER_TEST(SimplePower, "e^x", Sym::e() ^ Sym::var())
    PARSER_TEST(SimpleProduct, "2*arctg(x)", Sym::num(2) * Sym::arctan(Sym::var()))
    PARSER_TEST(SimpleDivision, "2/arctg(x)", Sym::num(2) / Sym::arctan(Sym::var()))

    PARSER_TEST(Sine, "sin(x)", Sym::sin(Sym::var()))
    PARSER_TEST(Cosine, "cos(x)", Sym::cos(Sym::var()))
    PARSER_TEST(Tangent, "tan(x)", Sym::tan(Sym::var()))
    PARSER_TEST(Tg, "tg(x)", Sym::tan(Sym::var()))
    PARSER_TEST(Cotangent, "cot(x)", Sym::cot(Sym::var()))
    PARSER_TEST(Ctg, "ctg(x)", Sym::cot(Sym::var()))

    PARSER_TEST(Sign, "sgn(x)", Sym::sgn(Sym::var()))
    PARSER_TEST(Absolute, "abs(x)", Sym::abs(Sym::var()))
    PARSER_TEST(ErrorFunction, "erf(x)", Sym::erf(Sym::var()))
    PARSER_TEST(SineIntegral, "Si(x)", Sym::si(Sym::var()))
    PARSER_TEST(CosineIntegral, "Ci(x)", Sym::ci(Sym::var()))
    PARSER_TEST(ExponentialIntegral, "Ei(x)", Sym::ei(Sym::var()))
    PARSER_TEST(LogarithmicIntegral, "li(x)", Sym::li(Sym::var()))

    PARSER_TEST(MultiplicationOverAddition, "x+4*pi", Sym::var() + (Sym::num(4) * Sym::pi()))
    PARSER_TEST(MultiplicationOverSubtraction, "x-4*pi", Sym::var() - (Sym::num(4) * Sym::pi()))
    PARSER_TEST(DivisionOverAddition, "x+4/pi", Sym::var() + (Sym::num(4) / Sym::pi()))
    PARSER_TEST(DivisionOverSubtraction, "x-4/pi", Sym::var() - (Sym::num(4) / Sym::pi()))
    PARSER_TEST(PowerOverMultiplication, "x*e^2", Sym::var() * (Sym::e() ^ Sym::num(2)))
    PARSER_TEST(PowerOverDivision, "x/e^2", Sym::var() / (Sym::e() ^ Sym::num(2)))
    PARSER_TEST(BracesOverPower, "2^(pi-x)", Sym::num(2) ^ (Sym::pi() - Sym::var()))
    PARSER_TEST(FunctionCallOverPower, "x^sin(e)", Sym::var() ^ Sym::sin(Sym::e()))
    PARSER_TEST(BracesOverFunctionCall, "ln(x-3*5)",
                Sym::ln(Sym::var() - Sym::num(3) * Sym::num(5)))
    PARSER_TEST(BracesOverLogarithmBase, "log_(x+1)(e)",
                Sym::log(Sym::var() + Sym::num(1), Sym::e()))

    PARSER_TEST(LeftAssociativityOfAddition, "x+1+e+pi",
                ((Sym::var() + Sym::num(1)) + Sym::e()) + Sym::pi())
    PARSER_TEST(LeftAssociativityOfMultiplication, "x*1*e*pi",
                ((Sym::var() * Sym::num(1)) * Sym::e()) * Sym::pi())
    PARSER_TEST(LeftAssociativityOfSutraction, "x-1-e-pi",
                ((Sym::var() - Sym::num(1)) - Sym::e()) - Sym::pi())
    PARSER_TEST(LeftAssociativityOfDivision, "x/1/e/pi",
                ((Sym::var() / Sym::num(1)) / Sym::e()) / Sym::pi())

    PARSER_TEST(RightAssociativityOfPower, "e^2^pi^x",
                Sym::e() ^ (Sym::num(2) ^ (Sym::pi() ^ Sym::var())))

    PARSER_TEST_ERR(ErrorOnUnrecognizedLetterSequence, "abcd")
    PARSER_TEST_ERR(ErrorOnFunctionNameWithoutArgument, "arcsin")
    PARSER_TEST_ERR(ErrorOnFunctionNameWithoutArgument2, "arcsin+1")

    PARSER_TEST(TrimSpaces, "  2 +    3  * sin   (  c )       ",
                Sym::num(2) + Sym::num(3) * Sym::sin(Sym::cnst("c")))
    PARSER_TEST_ERR(DoesNotRecognizeFunctionNameSplitWithSpaces, "t an(x)")

    PARSER_TEST(IntWithDifferential, "int x^2 dx", Sym::integral(Sym::var() ^ Sym::num(2)))
    PARSER_TEST(IntWithoutDifferential, "int x^2", Sym::integral(Sym::var() ^ Sym::num(2)))
    PARSER_TEST(IntegralWithDifferential, "integral x^2 dx",
                Sym::integral(Sym::var() ^ Sym::num(2)))
    PARSER_TEST(IntegralWithoutDifferential, "integral x^2",
                Sym::integral(Sym::var() ^ Sym::num(2)))
    PARSER_TEST(IntegrateWithDifferential, "integrate x^2 dx",
                Sym::integral(Sym::var() ^ Sym::num(2)))
    PARSER_TEST(IntegrateWithoutDifferential, "integrate x^2",
                Sym::integral(Sym::var() ^ Sym::num(2)))
    PARSER_TEST_ERR(ErrorOnDifferentialWithoutIntegral, "x^2 dx")

    PARSER_TEST(SimpleMultiplicationWithoutSign, "2x", Sym::num(2) * Sym::var())
    PARSER_TEST(MultiplicationOfLettersWithoutSign, "a b", Sym::cnst("a") * Sym::cnst("b"))
    PARSER_TEST(AdvancedMultiplicationWithoutSign, "31sin(7x y) a^2b^3 2^t(1+1)(x+cos(x))",
                Sym::num(31) * Sym::sin(Sym::num(7) * Sym::var() * Sym::cnst("y")) *
                    (Sym::cnst("a") ^ Sym::num(2)) * (Sym::cnst("b") ^ Sym::num(3)) *
                    (Sym::num(2) ^ Sym::cnst("t")) * (Sym::num(1) + Sym::num(1)) *
                    (Sym::var() + Sym::cos(Sym::var())))

    PARSER_TEST(NegationOfPower, "-x^2", -(Sym::var() ^ Sym::num(2)))
    PARSER_TEST(PowerOfNegation, "(-x)^2", (-Sym::var()) ^ Sym::num(2))
    PARSER_TEST(NegatedExponent, "x^-2", Sym::var() ^ (-Sym::num(2)))

    PARSER_TEST(MultiplicationWithNegationWithoutSign, "x (-x)", Sym::var() * (-Sym::var()))
    PARSER_TEST(SubtractionWithSpace, "x -x", Sym::var() - Sym::var())
    PARSER_TEST(MultiplicationWithNegation, "x*-x", Sym::var() * (-Sym::var()))
    PARSER_TEST(AdditionWithNegation, "x+--x", Sym::var() + (-(-Sym::var())))
    PARSER_TEST(NegationInSubtraction, "-x--x", (-Sym::var()) - (-Sym::var()))

    PARSER_TEST(FunctionSquared, "sin^2(x)", Sym::sin(Sym::var()) ^ Sym::num(2))
    PARSER_TEST(FunctionToAdvancedPower, "arcsin^(1+cos(x))^e^x((x+1)^2)^3",
                (Sym::arcsin((Sym::var() + Sym::num(1)) ^ Sym::num(2)) ^
                   ((Sym::num(1) + Sym::cos(Sym::var())) ^
                  (Sym::e() ^
                 Sym::var()))) ^
                    Sym::num(3))
    PARSER_TEST(LogarithmBase2Squared, "log_2^2(x)", Sym::log(Sym::num(2), Sym::var()) ^ Sym::num(2))
    PARSER_TEST(LogarithmBase4, "log_(2^2)(x)", Sym::log(Sym::num(2) ^ Sym::num(2), Sym::var()))
}
