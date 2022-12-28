#include <gtest/gtest.h>

#include "Parser/Scanner.cuh"

#define SCANNER_TEST(_name, _input, _expected_token, _expected_read_text) \
    TEST(ScannerTest, _name) { test_token_scanned(_input, _expected_token, _expected_read_text); } // NOLINT

#define SCANNER_TEST_ERR(_name, _input, _expected_read_text) \
    SCANNER_TEST(_name, _input, Token::Error, _expected_read_text)

namespace {
    void test_token_scanned(std::string input, Parser::Token expected_token,
                            std::string expected_read_text) {
        std::string read_text;
        Parser::Scanner scanner(input);
        const auto read_token = scanner.scan(read_text);
        EXPECT_EQ(read_token, expected_token);
        EXPECT_EQ(read_text, expected_read_text);
    }
}

namespace Test {
    using ::Parser::Token;

    SCANNER_TEST(ExpectEndFromEmptyString, "", Token::End, "<end>")
    SCANNER_TEST(ExpectPlus, "+23", Token::Plus, "+")
    SCANNER_TEST(ExpectMinus, "-a", Token::Minus, "-")
    SCANNER_TEST(ExpectDot, "*sin(x)", Token::Dot, "*")
    SCANNER_TEST(ExpectDash, "/3", Token::Dash, "/")
    SCANNER_TEST(ExpectCaret, "^", Token::Caret, "^")

    SCANNER_TEST(ExpectZeroInteger, "0", Token::Integer, "0")
    SCANNER_TEST(ExpectMultiDigitInteger, "2137", Token::Integer, "2137")

    SCANNER_TEST(ExpectLessThanOneDoubleWithLeadingZero, "0.3252", Token::Double, "0.3252")
    SCANNER_TEST(ExpectLessThanOneDoubleWithoutLeadingZero, ".3252", Token::Double, ".3252")
    SCANNER_TEST(ExpectLargerThanOneDouble, "420.420", Token::Double, "420.420")
    SCANNER_TEST(ExpectLargerThanOneDoubleWithoutDigitsAfterComma, "420.a", Token::Double, "420.")

    SCANNER_TEST(ExpectOneLetterSymbolicConstant, "a+2", Token::SymbolicConstant, "a")
    SCANNER_TEST(ExpectOneLetterSymbolicConstant2, "b^e", Token::SymbolicConstant, "b")
    SCANNER_TEST_ERR(ExpectErrorOnMultiLetterSymbolicConstant, "abcd", "abcd")

    SCANNER_TEST(ExpectVariable, "x", Token::Variable, "x")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterVariable, "xdd", "xdd")

    SCANNER_TEST(ExpectOpenBrace, "(1+2)", Token::OpenBrace, "(")
    SCANNER_TEST(ExpectCloseBrace, ")(1+2)", Token::CloseBrace, ")")
    SCANNER_TEST(ExpectUnderscore, "_f", Token::Underscore, "_")

    SCANNER_TEST(ExpectNeperConstant, "e", Token::E, "e")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterNeperConstant, "ee+", "ee")

    SCANNER_TEST(ExpectPi, "pi+2", Token::Pi, "pi")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterPi, "pie", "pie")

    SCANNER_TEST(ExpectInt, "int x dx", Token::Integral, "int")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterInt, "intt x dx", "intt")

    SCANNER_TEST(ExpectIntegral, "integral", Token::Integral, "integral")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterIntegral, "integrals", "integrals")

    SCANNER_TEST(ExpectIntegrate, "integrate", Token::Integral, "integrate")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterIntegrate, "integrateqq", "integrateqq")

    SCANNER_TEST(ExpectAbsoluteValue, "abs", Token::Abs, "abs")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterAbsoluteValue, "absolute(1)", "absolute")

    SCANNER_TEST(ExpectArcsine, "arcsin", Token::Asin, "arcsin")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterArcsine, "arcsine(1)", "arcsine")

    SCANNER_TEST(ExpectArccosine, "arccos", Token::Acos, "arccos")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterArccosine, "arccosasd", "arccosasd")

    SCANNER_TEST(ExpectArctan, "arctan", Token::Atan, "arctan")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterArctan, "arctangent", "arctangent")

    SCANNER_TEST(ExpectArctg, "arctg", Token::Atan, "arctg")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterArctg, "arctgnt", "arctgnt")

    SCANNER_TEST(ExpectArccot, "arccot+1", Token::Acot, "arccot")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterArccot, "arccotangent+1", "arccotangent")

    SCANNER_TEST(ExpectArcctg, "arcctg", Token::Acot, "arcctg")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterArcctg, "arcctgnt", "arcctgnt")

    SCANNER_TEST(ExpectCosine, "cos", Token::Cos, "cos")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterCosine, "cosine", "cosine")

    SCANNER_TEST(ExpectCotangent, "cot(x)", Token::Cot, "cot")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterCotangent, "cotangent(x)", "cotangent")

    SCANNER_TEST(ExpectCtg, "ctg", Token::Cot, "ctg")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterCtg, "ctgnt", "ctgnt")

    SCANNER_TEST(ExpectHyperbolicCosine, "cosh123", Token::Cosh, "cosh")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterHyperbolicCosine, "coshh123", "coshh")

    SCANNER_TEST(ExpectHyperbolicCotangent, "coth", Token::Coth, "coth")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterHyperbolicCotangent, "cothh", "cothh")

    SCANNER_TEST(ExpectHyperbolicCtg, "ctgh", Token::Coth, "ctgh")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterHyperbolicCtg, "ctghqw", "ctghqw")

    SCANNER_TEST(ExpectSign, "sgn(x)", Token::Sgn, "sgn")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterSign, "sgnn-23", "sgnn")

    SCANNER_TEST(ExpectSine, "sin(", Token::Sin, "sin")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterSine, "sine(", "sine")

    SCANNER_TEST(ExpectHyperbolicSine, "sinh_", Token::Sinh, "sinh")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterHyperbolicSine, "sinhyp_", "sinhyp")

    SCANNER_TEST(ExpectSquareRoot, "sqrt", Token::Sqrt, "sqrt")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterSquareRoot, "sqrtt", "sqrtt")

    SCANNER_TEST(ExpectTangent, "tan(", Token::Tan, "tan")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterTangent, "tangent(", "tangent")

    SCANNER_TEST(ExpectHyperbolicTangent, "tanh_", Token::Tanh, "tanh")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterHyperbolicTangent, "tanhyp_", "tanhyp")

    SCANNER_TEST(ExpectTg, "tg)(", Token::Tan, "tg")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterTg, "tgent)(", "tgent")

    SCANNER_TEST(ExpectHyperbolicTg, "tgh_t", Token::Tanh, "tgh")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterHyperbolicTg, "tghyp_t", "tghyp")

    SCANNER_TEST(ExpectNaturalLogarithm, "ln1", Token::Ln, "ln")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterNaturalLogarithm, "lne1", "lne")

    SCANNER_TEST(ExpectLogarithm, "log*2", Token::Log, "log")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterLogarithm, "logarithm*2", "logarithm")

    SCANNER_TEST(ExpectErrorFunction, "erf", Token::Erf, "erf")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterErrorFunction, "erfi", "erfi")

    SCANNER_TEST(ExpectSineIntegral, "Si1", Token::Si, "Si")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterSineIntegral, "Sin", "Sin")

    SCANNER_TEST(ExpectCosineIntegral, "Ci cos", Token::Ci, "Ci")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterCosineIntegral, "Cii cos", "Cii")

    SCANNER_TEST(ExpectExponentialIntegral, "Ei^2", Token::Ei, "Ei")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterExponentialIntegral, "Eixp", "Eixp")

    SCANNER_TEST(ExpectLogarithmicIntegral, "li", Token::Li, "li")
    SCANNER_TEST_ERR(ExpectErrorOnLettersAfterLogarithmicIntegral, "lint", "lint")
    SCANNER_TEST_ERR(LogarithmicIntegralCaseSensitive, "Li", "Li")

    SCANNER_TEST_ERR(ExpectErrorOnUnrecognizedSymbol, "%$$", "%")

    SCANNER_TEST(TrimSpacesBeforeToken, "   23", Token::Integer, "23")
    SCANNER_TEST(TrimSpacesAtEnd, "      ", Token::End, "<end>")

    SCANNER_TEST_ERR(ExpectErrorOnFunctionNameSplitWithSpaces, "co sh", "co")

}