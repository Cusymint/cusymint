#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "Evaluation/Integrate.cuh"
#include "Parser/Parser.cuh"
#include "Symbol/Integral.cuh"
#include "Symbol/InverseTrigonometric.cuh"
#include "Symbol/Macros.cuh"
#include "Symbol/Power.cuh"
#include "Symbol/Symbol.cuh"

#define DERIVATIVE_TEST(_name, _expression, _derivative) \
    TEST(Derivatives, _name) { test_derivative(_expression, _derivative); } // NOLINT

namespace Test {
    namespace {
        void test_derivative(std::vector<Sym::Symbol> expression,
                             std::vector<Sym::Symbol> expected_derivative) {
            std::vector<Sym::Symbol> help_space(Sym::EXPRESSION_MAX_SYMBOL_COUNT);
            std::vector<Sym::Symbol> derivative(Sym::EXPRESSION_MAX_SYMBOL_COUNT);

            const auto size = expression.data()->derivative_to(derivative.data());

            derivative.resize(size);

            EXPECT_EQ(size, derivative.data()->size())
                << "Tried to calculate derivative of:\n  " << expression.data()->to_string()
                << "\n but calculated size does not match actual size";

            expected_derivative.data()->simplify(help_space.data());
            derivative.data()->simplify(help_space.data());

            EXPECT_TRUE(
                Sym::Symbol::are_expressions_equal(*derivative.data(), *expected_derivative.data()))
                << "Tried to calculate derivative of:\n  " << expression.data()->to_string()
                << "\n  but got an unexpected result:\n  " << derivative.data()->to_string()
                << " <- got\n  " << expected_derivative.data()->to_string() << " <- expected\n";
        }

        void test_derivative(std::string expression, std::string expected_derivative) {
            test_derivative(Parser::parse_function(expression),
                            Parser::parse_function(expected_derivative));
        }
    }

    DERIVATIVE_TEST(Variable, "x", "1")
    DERIVATIVE_TEST(NumericConstant, "31.42", "0")
    DERIVATIVE_TEST(KnownConstant, "e", "0")
    DERIVATIVE_TEST(UnknownConstant, "c", "0")

    DERIVATIVE_TEST(AdvancedConstant, "sin(e)^tan(w+123-9+pi*arctan(2))*e+ln(tg(e^e^2^q))/8/sin(2)",
                    "0")

    DERIVATIVE_TEST(Arcsine, "arcsin(x)", "1/sqrt(1-x^2)")
    DERIVATIVE_TEST(Arccosine, Sym::arccos(Sym::var()),
                    -Sym::inv(Sym::sqrt(Sym::num(1) - (Sym::var() ^ Sym::num(2)))))
    DERIVATIVE_TEST(Arctangent, "arctan(x)", "1/(1+x^2)")
    DERIVATIVE_TEST(Arccotangent, Sym::arccot(Sym::var()),
                    -Sym::inv(Sym::num(1) + (Sym::var() ^ Sym::num(2))))

    DERIVATIVE_TEST(Ln, "ln(x)", "1/x")

    DERIVATIVE_TEST(Sine, "sin(x)", "cos(x)")
    DERIVATIVE_TEST(Cosine, "cos(x)", "-sin(x)")
    DERIVATIVE_TEST(Tangent, "tg(x)", "1/cos(x)^2")
    DERIVATIVE_TEST(Cotangent, Sym::cot(Sym::var()), -Sym::inv(Sym::sin(Sym::var()) ^ Sym::num(2)))

    DERIVATIVE_TEST(Negation, "-cos(x)", "sin(x)")
    DERIVATIVE_TEST(Reciprocal, Sym::inv(Sym::var()), -Sym::inv(Sym::var() ^ Sym::num(2)));

    DERIVATIVE_TEST(Addition, "sin(x)+ln(x)", "cos(x)+1/x")
    DERIVATIVE_TEST(ProductWithConstant, "ln(1+tg(e))*cos(x)", "ln(1+tg(e))*(-sin(x))")
    DERIVATIVE_TEST(Product, "x*ln(x)", "ln(x)+1")

    DERIVATIVE_TEST(EToX, "e^x", "e^x")
    DERIVATIVE_TEST(EToXTower, "e^e^e^x", "e^e^e^x*e^e^x*e^x")
    DERIVATIVE_TEST(Monomial, "x^(4+e)", "(4+e)*x^(3+e)")
    DERIVATIVE_TEST(Power, "x^x", "x^x*(1+ln(x))")

    DERIVATIVE_TEST(Exponential, "2^x", "2^x*ln(2)")
    DERIVATIVE_TEST(Logarithm, "log_3(x)", "1/x/ln(3)")

    DERIVATIVE_TEST(AdvancedExpressionWithArctg, "1/sin(x)+arctg(sqrt(1+x^2))",
                    "cos(x)*(-(1/sin(x)^2))+1/(2+x^2)*(1+x^2)^-0.5*x")
    DERIVATIVE_TEST(AdvancedExpressionWithLogCubed, "4*ln((sin(2*x)+1/(3*x+2))^2)^3",
                    "24*ln((sin(2*x)+1/(3*x+2))^2)^2/(sin(2*x)+1/(3*x+2))^2*(sin(2*x)+1/"
                    "(3*x+2))*(2*cos(2*x)+3*(-(1/(3*x+2)^2)))")

    DERIVATIVE_TEST(Polynomial, "x^6+2*x^5+9*x^4+2*x^3+x^2+9*x+1",
                    "6*x^5+10*x^4+36*x^3+6*x^2+2*x+9")
    DERIVATIVE_TEST(RationalFunction, "(p*x^2+q*x+r)/(a*x+b)",
                    "(a*(p*x^2+q*x+r))*(-(1/(a*x+b)^2))+(2*p*x+q)/(a*x+b)")
    DERIVATIVE_TEST(DividingFunctionByFunction, "sin(x)/x", "sin(x)*(-(1/x^2))+cos(x)/x")
}