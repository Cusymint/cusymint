#include "Performance.cuh"
#include "Symbol/Integral.cuh"
#include "Symbol/SymbolType.cuh"
#include <cstddef>
#include <fmt/core.h>
#include <string>
#include <vector>

namespace Performance {
    namespace {
        std::string exec_and_read_output(const std::string cmd) {
            // stolen from SolverProcessManager.cu
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
            std::string expression = integral;
            capitalize_substring(expression, "e");
            capitalize_substring(expression, "pi");
            capitalize_substring(expression, "sqrt");

            return fmt::format(
                R"(wolframscript -code 'expr=ToExpression["{}",TraditionalForm];Integrate[expr,x];{{t,b}}=AbsoluteTiming[Integrate[expr,x]];t')",
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
                R"(python3 -c 'from sympy import *;x=Symbol("x");t=Symbol("t");integrate({},x);print(utilities.timeutils.timed(lambda:integrate({},x))[1])')",
                expression, expression);
        }

        std::string make_matlab_command(const std::string& integral) {
            std::string expression = integral;
            capitalize_substring(expression, "e");
            replace(expression, "E", "exp(sym(1))");
            replace(expression, "arctan", "atan");
            replace(expression, "arccos", "acos");
            replace(expression, "arcsin", "asin");
            replace(expression, "sgn", "sign");

            return fmt::format(
                R"(matlab -batch 'syms x z;f=@()int({},x);f();fprintf("%f\n",timeit(f));')",
                expression);
        }

        std::string make_mathematica_command_batch(const std::vector<std::string>& integrals) {
            std::string result = "wolframscript -code \'";
            for (const auto& integral : integrals) {
                std::string expression = integral;
                capitalize_substring(expression, "e");
                capitalize_substring(expression, "pi");
                capitalize_substring(expression, "sqrt");
                result += fmt::format("{{t,b}}=AbsoluteTiming[Integrate[ToExpression[\"{}\","
                                      "TraditionalForm],x]];Print[t];ClearSystemCache[];",
                                      expression);
            }

            return result + "\'";
        }

        std::string make_sympy_command_batch(const std::vector<std::string>& integrals) {
            std::string result = R"(python3 -c 'from sympy import *;x=Symbol("x");t=Symbol("t");)";
            for (const auto& integral : integrals) {
                std::string expression = integral;
                capitalize_substring(expression, "e");
                replace(expression, "^", "**");
                replace(expression, "arctan", "atan");
                replace(expression, "arccos", "acos");
                replace(expression, "arcsin", "asin");
                replace(expression, "sgn", "sign");
                result += fmt::format(
                    "print(utilities.timeutils.timed(lambda:integrate({},x))[1]);", expression);
            }

            return result + "\'";
        }

        std::string make_matlab_command_batch(const std::vector<std::string>& integrals) {
            std::string result = R"(matlab -batch 'syms x t;)";
            for (const auto& integral : integrals) {
                std::string expression = integral;
                capitalize_substring(expression, "e");
                replace(expression, "E", "exp(sym(1))");
                replace(expression, "arctan", "atan");
                replace(expression, "arccos", "acos");
                replace(expression, "arcsin", "asin");
                replace(expression, "sgn", "sign");
                replace(expression, "ln", "log");

                // result += fmt::format(R"(f=@()int({},x);fprintf("%f\n",timeit(f));)",
                // expression);
                result += fmt::format(R"(tic;int({},x);t=toc;fprintf("%f\n",t);)", expression);
            }

            return result + "\'";
        }
    }

    void do_not_print_results(const std::string& integral_str, const double& cusymint_seconds,
                              bool cusymint_success, const std::string& mathematica_result,
                              const std::string& sympy_result, const std::string& matlab_result) {}

    void print_human_readable_results(const std::string& integral_str,
                                      const double& cusymint_seconds, bool cusymint_success,
                                      const std::string& mathematica_result,
                                      const std::string& sympy_result,
                                      const std::string& matlab_result) {
        printf("%s:\n"
               "  Cusymint time:    %f%s\n"
               "  Mathematica time: %s\n"
               "  Sympy time:       %s\n"
               "  Matlab time:      %s\n\n",
               integral_str.c_str(), cusymint_seconds, cusymint_success ? "" : " (failure)",
               mathematica_result.c_str(), sympy_result.c_str(), matlab_result.c_str());
    }

    void print_csv_results(const std::string& integral_str, const double& cusymint_seconds,
                           bool cusymint_success, const std::string& mathematica_result,
                           const std::string& sympy_result, const std::string& matlab_result) {
        printf("%s;%s;%f;%s;%s;%s\n", integral_str.c_str(), cusymint_success ? "TRUE" : "FALSE",
               cusymint_seconds, mathematica_result.c_str(), sympy_result.c_str(),
               matlab_result.c_str());
    }

    void print_csv_headers() {
        printf("integral;solved_by_cusymint;cusymint_time[s];mathematica_time[s];sympy_time[s];"
               "matlab_time[s]\n");
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
        auto matlab_result = exec_and_read_output(make_matlab_command(integral[1].to_string()));

        print_results(integral_str, cusymint_seconds, result.has_value(), mathematica_result,
                      sympy_result, matlab_result);
    }

    void test_cuda_and_print_commands(const std::vector<std::string>& integrals) {
        printf("Computing on CUDA...\n");

        for (int i = 0; i < integrals.size(); ++i) {
            Sym::Integrator integrator;
            const auto integral = Sym::integral(Parser::parse_function(integrals[i]));
            const clock_t start = clock();
            const auto result = integrator.solve_integral(integral);
            const clock_t end = clock();

            fmt::print("{}{}\n", static_cast<double>(end - start) / CLOCKS_PER_SEC,
                       result.has_value() ? "" : " (failure)");
        }

        printf("%s\n", make_mathematica_command_batch(integrals).c_str());
        printf("%s\n", make_sympy_command_batch(integrals).c_str());
        printf("%s\n", make_matlab_command_batch(integrals).c_str());
    }

    void test_memory_occupance(const std::vector<std::string>& integrals) {
        printf("Memory occupance:\n");
        for (int i = 0; i < integrals.size(); ++i) {
            printf("  Integral %s:\n", integrals[i].c_str());

            size_t initial_mem;
            size_t after_init;

            cudaMemGetInfo(&initial_mem, nullptr);

            Sym::Integrator integrator;

            cudaMemGetInfo(&after_init, nullptr);
            initial_mem -= after_init;

            size_t total;
            const auto integral = Sym::integral(Parser::parse_function(integrals[i]));

            const size_t usage = integrator.memory_usage_for_integral(integral, total);

            printf("    %10lu/%10lu B used (%f%%)\n", usage + initial_mem, total,
                   static_cast<double>(usage + initial_mem) / total * 100);
        }
    }

    void test_history_cpu_memory_occupance(const std::vector<std::string>& integrals) {
        printf("Memory occupance for history:\n");
        for (int i = 0; i < integrals.size(); ++i) {
            printf("  Integral %s:\n", integrals[i].c_str());

            Sym::Integrator integrator;
            Sym::ComputationHistory history;
            const auto integral = Sym::integral(Parser::parse_function(integrals[i]));

            const auto result = integrator.solve_integral_with_history(integral, history);

            printf("    %10luB used%s\n", history.get_memory_occupance_in_bytes(), result.has_value() ? "" : " (failure)");
        }
    }

    void test_performance(const std::vector<std::string>& integrals, PrintRoutine print_results) {
        std::vector<double> cusymint_seconds_vector(integrals.size());
        std::vector<bool> cusymint_results_vector(integrals.size());
        std::vector<std::string> mathematica_results_vector(integrals.size());
        std::vector<std::string> sympy_results_vector(integrals.size());
        std::vector<std::string> matlab_results_vector(integrals.size());

        printf("Computing on CUDA...\n");

        for (int i = 0; i < integrals.size(); ++i) {
            Sym::Integrator integrator;
            const auto integral = Sym::integral(Parser::parse_function(integrals[i]));
            const clock_t start = clock();
            const auto result = integrator.solve_integral(integral);
            const clock_t end = clock();

            cusymint_seconds_vector[i] = static_cast<double>(end - start) / CLOCKS_PER_SEC;
            cusymint_results_vector[i] = result.has_value();
        }

        printf("Computing on Mathematica...\n");
        auto mathematica_result = exec_and_read_output(make_mathematica_command_batch(integrals));
        // Warning: computing na integral which requires many substitutions on SymPy may hang your
        // computer and is extremely slow!
        printf("Computing on SymPy...\n");
        auto sympy_result = exec_and_read_output(make_sympy_command_batch(integrals));
        printf("Computing on MATLAB...\n");
        auto matlab_result = exec_and_read_output(make_matlab_command_batch(integrals));

        size_t mc_idx = 0;
        size_t sp_idx = 0;
        size_t ml_idx = 0;
        for (int i = 0; i < integrals.size(); ++i) {
            const auto mc_inc = mathematica_result.find('\n', mc_idx);
            if (mc_inc != std::string::npos) {
                mathematica_results_vector[i] = mathematica_result.substr(mc_idx, mc_inc - mc_idx);
                mc_idx = mc_inc + 1;
            }

            const auto sp_inc = sympy_result.find('\n', sp_idx);
            if (sp_inc != std::string::npos) {
                sympy_results_vector[i] = sympy_result.substr(sp_idx, sp_inc - sp_idx);
                sp_idx = sp_inc + 1;
            }

            const auto ml_inc = matlab_result.find('\n', ml_idx);
            if (ml_inc != std::string::npos) {
                matlab_results_vector[i] = matlab_result.substr(ml_idx, ml_inc - ml_idx);
                ml_idx = ml_inc + 1;
            }
        }

        if (print_results == print_csv_results) {
            print_csv_headers();
        }

        for (int i = 0; i < integrals.size(); ++i) {
            print_results(integrals[i], cusymint_seconds_vector[i], cusymint_results_vector[i],
                          mathematica_results_vector[i], sympy_results_vector[i],
                          matlab_results_vector[i]);
        }
    }

}