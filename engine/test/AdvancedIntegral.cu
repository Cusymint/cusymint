#include "IntegralCommons.cuh"

#include <cctype>
#include <cstddef>
#include <ctime>
#include <fmt/core.h>
#include <string>

#include "Evaluation/Integrator.cuh"
#include "Parser/Parser.cuh"
#include "Simplify.cuh"
#include "Symbol/Macros.cuh"
#include "Symbol/Symbol.cuh"

#define ADVANCED_INTEGRAL_TEST_CUDA_ONLY(_name, _integral, _expected_result) \
    INTEGRAL_TEST(AdvancedIntegral, _name, _integral, _expected_result)

#define ADVANCED_INTEGRAL_TEST_TIME(_name, _integral, _expected_result) \
    TEST_F(AdvancedIntegral, _name) { test_integral_with_external_tools(_integral); }

#define ADVANCED_INTEGRAL_TEST(_name, _integral, _expected_result) \
    ADVANCED_INTEGRAL_TEST_TIME(_name, _integral, _expected_result)

namespace Test {
    namespace {
        std::string exec_and_read_output(const std::string cmd) {
            // sciagniete z SolverProcessManager.cu
            std::array<char, 4096> buffer{};
            std::string result;
            std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
            if (!pipe) {
                throw std::runtime_error("popen() failed!");
            }
            while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
                result += buffer.data();
            }
            return result;
        }

        void capitalize_substring(std::string& string, const std::string& what) {
            if (string.size() < what.size()) {
                return;
            }
            for (size_t i = 0; i <= string.size() - what.size(); ++i) {
                if (string.substr(i, what.size()) == what &&
                    (i == 0 || !std::isalpha(string[i - 1])) &&
                    (i == string.size() - what.size() || !std::isalpha(string[i + what.size()]))) {
                    string[i] = string[i] - 'a' + 'A';
                }
            }
        }

        void replace(std::string& string, const std::string& from, const std::string& to) {
            if (string.size() < from.size()) {
                return;
            }
            for (ssize_t i = 0; i <= string.size() - from.size(); ++i) {
                if (string.substr(i, from.size()) == from) {
                    string = string.substr(0, i) + to + string.substr(i + from.size());
                }
            }
        }

        std::string make_mathematica_command(const std::string& integral) {
            // /media/cmonbrug/DATA/mathematica/Executables/wolframscript -code
            // 'expr=ToExpression["sin(Pi)+x",TraditionalForm];{t,b}=AbsoluteTiming[Integrate[expr,x]];t'
            std::string expression = integral;
            capitalize_substring(expression, "e");
            capitalize_substring(expression, "pi");
            capitalize_substring(expression, "sqrt");
            return fmt::format(
                R"(wolframscript -code 'expr=ToExpression["{}",TraditionalForm];{{t,b}}=AbsoluteTiming[Integrate[expr,x]];t')",
                expression);
        }

        std::string make_sympy_command(const std::string& integral) {
            std::string expression = integral;
            capitalize_substring(expression, "e");
            replace(expression, "^", "**");
            replace(expression, "arctan", "atan");
            replace(expression, "arccos", "acos");
            replace(expression, "arcsin", "asin");
            replace(expression, "sgn", "sign");
            //fmt::print(R"(Call: python3 -c 'from sympy import *;x=Symbol("x");print(utilities.timeutils.timed(lambda:integrate({},x))[1])')" "\n",
            //    expression);
            return fmt::format(
                R"(python3 -c 'from sympy import *;x=Symbol("x");t=Symbol("t");print(utilities.timeutils.timed(lambda:integrate({},x))[1])')",
                expression);
        }

        void test_integral_with_external_tools(const std::string& integral_str) {
            const auto integral = Sym::integral(Parser::parse_function(integral_str));

            Sym::Integrator integrator;

            const clock_t start = clock();
            const auto result = integrator.solve_integral(integral);
            const clock_t end = clock();

            const double cusymint_seconds = static_cast<double>(end - start) / CLOCKS_PER_SEC;

            auto mathematica_result = exec_and_read_output(make_mathematica_command(integral_str));
            auto sympy_result = exec_and_read_output(make_sympy_command(integral[1].to_string()));

            printf("Cusymint time:    %f\n"
                   "Mathematica time: %s\n"
                   "Sympy time:       %s\n",
                   cusymint_seconds, mathematica_result.c_str(), sympy_result.c_str());
        }
    }

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
    // ADVANCED_INTEGRAL_TEST(NegativeExponentByParts, "x e^(-x)", "-e^(-x)(x+1)") // linear
    // substitution for nested-vars-only required
    ADVANCED_INTEGRAL_TEST(CosineByParts2, "x^2cos(x)", "x^2sin(x)+2x cos(x)-2sin(x)")
    ADVANCED_INTEGRAL_TEST(LogarithmTimesRoot, "sqrt(x)ln(x)", "2/3x^(3/2)(ln(x)-2/3)")
    ADVANCED_INTEGRAL_TEST(LogarithmDividedByPower, "ln(x)/x^4", "-1/(3x^3)(ln(x)+1/3)")

    ADVANCED_INTEGRAL_TEST(VeryLongPolynomial, "(2x-1)^20", "1/42(2x-1)^21+1/42")
    ADVANCED_INTEGRAL_TEST(CosineOfLinearFunction, "cos(3x-1)", "sin(3x-1)/3")
    // ADVANCED_INTEGRAL_TEST(RationalWithSubstitution, "2x/(5x^2+1)", "1/5ln(5x^2+1)") //
    // substitution t=x^2+a required
    ADVANCED_INTEGRAL_TEST(SimpleRationalWithExponent, "e^x/(3+e^x)", "ln(3+e^x)")
    // ADVANCED_INTEGRAL_TEST(ExponentWithSquare, "x*e^(-x^2)", "-1/2e^(-x^2)") // substitution
    // t=x^2+a required ADVANCED_INTEGRAL_TEST(ExponentOfInverse, "e^(1/x)/x^2", "-e^(1/x)") //
    // substitution t=ax^b+c required (dx = 1/b((t-c)/a)^((1-b)/b))
    ADVANCED_INTEGRAL_TEST(Tangent, "tg(x)", "-ln(cos(x))")
    ADVANCED_INTEGRAL_TEST(TangentDividedByCosSquared, "tg(x)/cos^2(x)", "tg^2(x)/2")
    ADVANCED_INTEGRAL_TEST(CosTimesExpSine, "cos(x)e^sin(x)", "e^sin(x)")
    ADVANCED_INTEGRAL_TEST(CosineDividedByRootOfLinSine, "cos(x)/sqrt(1+sin(x))", "2sqrt(1+sin(x))")
    // ADVANCED_INTEGRAL_TEST(TangentDerivativeWithSubstitution, "x^3/cos^2(x^4)", "tg(x^4)/4") //
    // substitution t=ax^b+c required (dx = 1/b((t-c)/a)^((1-b)/b))
    ADVANCED_INTEGRAL_TEST(ExponentOfLinear, "6^(1-x)", "-6^(1-x)/ln(6)")
    // ADVANCED_INTEGRAL_TEST(ArctanSubtitution, "1/(1+x^2)/arctan(x)", "ln(arctan(x))") //
    // substitution t=arctan(x) required

    // task 4 contains examples which require nested integrals (not implemeneted)

    // harder ones from WUT MiNI faculty
    // https://pages.mini.pw.edu.pl/~dembinskaa/www/?download=Inf_I_cw9_2022_23.pdf, tasks 1-3
    ADVANCED_INTEGRAL_TEST(SimpleRationalWithSquare, "(sqrt(x)-2)^2/x^2", "-4/x+8/sqrt(x)+ln(x)")
    ADVANCED_INTEGRAL_TEST(SineCosineSquaredInDenominator, "1/(sin^2(x)cos^2(x))", "-1/tg(x)+tg(x)")

    // ADVANCED_INTEGRAL_TEST(LogarithmDividedByX, "ln^5(x)/x", "ln^6(x)/6") // substitution t=ln(x)
    // or cyclic integrals required ADVANCED_INTEGRAL_TEST(ArctangentWithX3Substitution,
    // "x^3/(x^8+1)", "arctg(x^4)/4") // substitution t=x^4 required
    // ADVANCED_INTEGRAL_TEST(LogarithmMultipliedByX, "x^n*ln(x)",
    // "x^(n+1)ln(x)/(n+1)-x^(n+1)/(n+1)^2") // non-numeric powers integration by parts required
    ADVANCED_INTEGRAL_TEST(ArctangentMultipliedByX, "x*arctg(x)", "x^2/2*arctg(x)-x/2+arctg(x)/2")
    // ADVANCED_INTEGRAL_TEST(Arcsine, "arcsin(x)", "x*arcsin(x)+2*sqrt(1-x^2)") // substitution
    // t=x^2+a required ADVANCED_INTEGRAL_TEST(LogarithmWithX2Substitution, "x*ln(x^2+1)",
    // "(x^2+1)ln(x^2+1)-x^2-1") // substitution t=x^2+a required
    // ADVANCED_INTEGRAL_TEST(CyclicIntegral1, "e^x*sin(2x)", "e^x*sin(2x)/5 - e^x*2cos(2x)/5") //
    // cyclic integrals required ADVANCED_INTEGRAL_TEST(CyclicIntegral2, "sin(ln(x))",
    // "-x/2(cos(ln(x))-sin(ln(x)))") // cyclic integrals required
    // ADVANCED_INTEGRAL_TEST(CotangentDividedByLogOfSine, "ctg(x)/ln(sin(x))", "ln(ln(sin(x)))") //
    // substitution t=ln(x) required ADVANCED_INTEGRAL_TEST(InvertedCosH, "1/(e^x+e^-x)",
    // "arctg(e^x)") // extract_function required

    ADVANCED_INTEGRAL_TEST(Absolute, "abs(x)", "x*abs(x)/2") // abs required
    ADVANCED_INTEGRAL_TEST(MaxOfOneAndSquare, "(abs(1-x^2)+1+x^2)/2",
                           "(sgn(1-x^2)(x-x^3/3)+x+x^3/3)/2") // abs required

    // tasks 5-7 involve rational integrals

    // https://pages.mini.pw.edu.pl/~dembinskaa/www/?download=Inf_I_PowtKol3_2022-2023.pdf, task 11
    ADVANCED_INTEGRAL_TEST(TangentSquared, "tg^2(x)", "tg(x)-arctg(tg(x))") // may sometimes fail
    ADVANCED_INTEGRAL_TEST(LongPolynomial, "(2x-3)^10", "1/22(2x-3)^11+3^11/22")
    // ADVANCED_INTEGRAL_TEST(SquareRootInDenominator, "1/(2+sqrt(x))", "2sqrt(x)-4ln(sqrt(x)+2)")
    // // substitution t=sqrt(x)+a required ADVANCED_INTEGRAL_TEST(InvertedCosH2, "1/cosh(x)",
    // "2arctg(e^x)") // extract_funcion required
    ADVANCED_INTEGRAL_TEST(Sine5Cosine, "sin^5(x)cos(x)", "sin^6(x)/6")
    // ADVANCED_INTEGRAL_TEST(ExpressionWithSquareInDenominator, "x/(x^2-1)^(3/2)",
    // "-1/sqrt(x^2-1)") // substitution t=ax^b+c required
    // ADVANCED_INTEGRAL_TEST(ArcsineWithDerivative, "arcsin^2(x)/sqrt(1-x^2)", "arcsin^3(x)/3") //
    // substitution t=arcsin(x) required ADVANCED_INTEGRAL_TEST(RootInDenominator, "1/(x^(1/3)+1)",
    // "3/2*x^(2/3)-3x^(1/3)+3ln(x^(1/3)+1)") // substitution t=ax^b+c required (dx =
    // 1/b((t-c)/a)^((1-b)/b))
    ADVANCED_INTEGRAL_TEST(Logarithm, "ln(x)", "x ln(x) - x")
    ADVANCED_INTEGRAL_TEST(XTimesCosine, "x*cos(x)", "x*sin(x)+cos(x)")
    // ADVANCED_INTEGRAL_TEST(XTimesExponential, "x^2e^(1-x)", "-(x^2+2x+2)e^(1-x)") // linear
    // substitution for nested-vars-only required ADVANCED_INTEGRAL_TEST(SquareTimesExponential,
    // "8x^2e^(4-x^3)", "-8/3e^(4-x^3)") // substitution t=x^2+a required

    // ADVANCED_INTEGRAL_TEST(SineSquared, "sin^2(x)", "x/2-sin(2x)/4") // rational integrals
    // required
    ADVANCED_INTEGRAL_TEST(Sine5, "sin^5(x)", "-cos^5(x)/5+2/3cos^3(x)-cos(x)")
    ADVANCED_INTEGRAL_TEST(Sine4Cos3, "sin^4(x)cos^3(x)", "sin^5(x)/5-sin^7(x)/7")
    // ADVANCED_INTEGRAL_TEST(Sine4Cos2, "sin^4(x)cos^2(x)",
    // "1/6sin^5(x)cos(x)+1/24sin^3(x)cos(x)-1/32sin(x)cos(x)+1/16x") // rational integrals required
    // ADVANCED_INTEGRAL_TEST(Cos3XCos5X, "cos(3x)cos(5x)", "1/16sin(8x)+1/4sin(2x)") // more
    // trigonometric identitied required

    // own tests
    ADVANCED_INTEGRAL_TEST(ETower, "e^x*e^e^x*e^e^e^x*e^e^e^e^x*e^e^e^e^e^x*e^e^e^e^e^e^x",
                           "e^e^e^e^e^e^x")
}