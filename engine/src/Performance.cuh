#ifndef PERFORMACNE_CUH
#define PERFORMANCE_CUH

#include <cctype>
#include <cstddef>
#include <ctime>
#include <fmt/core.h>
#include <string>

#include "Evaluation/Integrator.cuh"
#include "Parser/Parser.cuh"
#include "Symbol/Macros.cuh"
#include "Symbol/Symbol.cuh"

namespace Performance {

    using PrintRoutine = void (*)(const std::string& integral_str, const double& cusymint_seconds,
                                  bool cusymint_success, const std::string& mathematica_result,
                                  const std::string& sympy_result, const std::string& matlab_result);

    void do_not_print_results(const std::string& integral_str, const double& cusymint_seconds,
                              bool cusymint_success, const std::string& mathematica_result,
                              const std::string& sympy_result, const std::string& matlab_result);
    void print_human_readable_results(const std::string& integral_str,
                                      const double& cusymint_seconds, bool cusymint_success,
                                      const std::string& mathematica_result,
                                      const std::string& sympy_result, const std::string& matlab_result);

    void print_csv_results(const std::string& integral_str, const double& cusymint_seconds,
                           bool cusymint_success, const std::string& mathematica_result,
                           const std::string& sympy_result, const std::string& matlab_result);

    void test_with_other_solutions(const std::string& integral_str,
                                   PrintRoutine print_results = print_human_readable_results);
}

#endif