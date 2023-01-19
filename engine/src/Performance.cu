#include "Performance.cuh"

namespace Performance {
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

            return fmt::format(
                R"(python3 -c 'from sympy import *;x=Symbol("x");t=Symbol("t");print(utilities.timeutils.timed(lambda:integrate({},x))[1])')",
                expression);
        }
    }

    void do_not_print_results(const std::string& integral_str, const double& cusymint_seconds,
                              bool cusymint_success, const std::string& mathematica_result,
                              const std::string& sympy_result) {}

    void print_human_readable_results(const std::string& integral_str,
                                      const double& cusymint_seconds, bool cusymint_success,
                                      const std::string& mathematica_result,
                                      const std::string& sympy_result) {
        printf("%s:\n"
               "  Cusymint time:    %f%s\n"
               "  Mathematica time: %s\n"
               "  Sympy time:       %s\n",
               integral_str.c_str(), cusymint_seconds, cusymint_success ? "" : " (failure)",
               mathematica_result.c_str(), sympy_result.c_str());
    }

    void print_csv_results(const std::string& integral_str, const double& cusymint_seconds,
                           bool cusymint_success, const std::string& mathematica_result,
                           const std::string& sympy_result) {
        printf("%s;%s;%f;%s;%s\n", integral_str.c_str(), cusymint_success ? "TRUE" : "FALSE",
               cusymint_seconds, mathematica_result.c_str(), sympy_result.c_str());
    }

    void test_with_other_solutions(const std::string& integral_str, PrintRoutine print_results) {
        const auto integral = Sym::integral(Parser::parse_function(integral_str));

        Sym::Integrator integrator;

        const clock_t start = clock();
        const auto result = integrator.solve_integral(integral);
        const clock_t end = clock();

        const double cusymint_seconds = static_cast<double>(end - start) / CLOCKS_PER_SEC;

        auto mathematica_result = exec_and_read_output(make_mathematica_command(integral_str));
        auto sympy_result = exec_and_read_output(make_sympy_command(integral[1].to_string()));

        print_results(integral_str, cusymint_seconds, result.has_value(), mathematica_result, sympy_result);
    }
}