#include <cstdlib>

#include <iostream>
#include <optional>
#include <vector>

#include <fmt/core.h>

#include "Evaluation/Integrator.cuh"
#include "Evaluation/StaticFunctions.cuh"

#include "Performance.cuh"
#include "Symbol/Constants.cuh"
#include "Symbol/ExpressionArray.cuh"
#include "Symbol/Integral.cuh"
#include "Symbol/Symbol.cuh"
#include "Symbol/Variable.cuh"

#include "Parser/Parser.cuh"

#include "Utils/CompileConstants.cuh"

/*
 * @brief Creates a `std::string` representing expression of type `e^x * e^e^x * ... * e^e^...^e^x`,
 * which is made of `n` factors.
 *
 * @param `n` - number of factors in created expression.
 *
 * @return Created string. If `n==0`, function returns `"1"`.
 */
std::string e_tower(size_t n) {
    if (n == 0) {
        return "1";
    }
    std::string res = "e^x";
    for (int i = 2; i <= n; ++i) {
        res += "*";
        for (int j = 1; j <= i; ++j) {
            res += "e^";
        }
        res += "x";
    }
    return res;
}

void cuda_warmup() {
    Sym::Integrator integrator;
    integrator.solve_integral(Sym::integral(Sym::var() + Sym::sin(Sym::var())));
}

void test_performance(const std::vector<std::string>& integrals) {
    for (auto integral_str : integrals) {
        Performance::test_with_other_solutions(integral_str,
                                               Performance::print_human_readable_results);
    }
}

int main() {
    if constexpr (Consts::DEBUG) {
        fmt::print("Running in debug mode\n");
    }

    Sym::Static::init_functions();

    cuda_warmup();

    // test_performance({
    //     "x^4-5x+3",
    //     "(1-x^3)^2",
    //     "x^3t^2",
    //     "x^(2/3)",
    //     "1/x^(2/3)",
    //     "(x^3-3x^2+1)/x^5",
    //     "(x^(2/3)-sqrt(x))/x^(1/3)",
    //     "2e^x+3*5^x",
    //     "x cos(x)",
    //     "x e^x",
    //     "x^2e^x",
    //     "x^2cos(x)",
    //     "sqrt(x)ln(x)",
    //     "ln(x)/x^4",
    //     "(2x-1)^20",
    //     "cos(3x-1)",
    //     "e^x/(3+e^x)",
    //     "tg(x)",
    //     "tg(x)/cos^2(x)",
    //     "cos(x)e^sin(x)",
    //     "cos(x)/sqrt(1+sin(x))",
    //     "6^(1-x)",
    //     "(sqrt(x)-2)^2/x^2",
    //     "1/(sin^2(x)cos^2(x))",
    //     "x*arctg(x)",
    //     "abs(x)",
    //     "(abs(1-x^2)+1+x^2)/2",
    //     "tg^2(x)",
    //     "(2x-3)^10",
    //     "sin^5(x)cos(x)",
    //     "ln(x)",
    //     "x*cos(x)",
    //     "x^2e^(1-x)",
    //     "sin^5(x)",
    //     "sin^4(x)cos^3(x)",
    //     "e^x*e^e^x*e^e^e^x*e^e^e^e^x*e^e^e^e^e^x*e^e^e^e^e^e^x",
    // });

    const auto integral = Sym::integral(Parser::parse_function("tg(x)^2"));

    fmt::print("Trying to solve an integral: {}\n", integral.data()->to_tex());

    Sym::Integrator integrator;
    Sym::ComputationHistory history;
    const auto solution = integrator.solve_integral_with_history(integral, history);

    if (solution.has_value()) {
        fmt::print("Success! Solution:\n{} + C\n", solution.value().data()->to_tex());

        fmt::print("\nComputation steps:\n\n");

        for (const auto& step: history.get_tex_history()) {
            fmt::print("  {}\\\\\n\n", step);
        }
    }
    else {
        fmt::print("No solution found\n");
    }
}
