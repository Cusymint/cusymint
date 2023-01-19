#include <cstdlib>

#include <fmt/format.h>
#include <iostream>
#include <optional>
#include <string>
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

int main() {
    if constexpr (Consts::DEBUG) {
        fmt::print("Running in debug mode\n");
    }

    Sym::Static::init_functions();

    fmt::print("First two integrals are for warmup!\n");

    std::vector<std::string> integrals = {
        "x+sin(x)",
        "e^x",
        "x^4-5*x+3",
        "(1-x^3)^2",
        "x^3*t^2",
        "x^(2/3)",
        "1/x^(2/3)",
        "(x^3-3*x^2+1)/x^5",
        "(x^(2/3)-sqrt(x))/x^(1/3)",
        "2*e^x+3*5^x",
        "x*cos(x)",
        "x*e^x",
        "x^2*e^x",
        "x^2*cos(x)",
        "sqrt(x)*ln(x)",
        "ln(x)/x^4",
        "(2*x-1)^20",
        "cos(3*x-1)",
        "e^x/(3+e^x)",
        "tan(x)",
        "tan(x)/cos(x)^2",
        "cos(x)*e^sin(x)",
        "cos(x)/sqrt(1+sin(x))",
        "6^(1-x)",
        "(sqrt(x)-2)^2/x^2",
        "1/(sin(x)^2*cos(x)^2)",
        "x*arctan(x)",
        "abs(x)",
        "(abs(1-x^2)+1+x^2)/2",
        "tan(x)^2",
        "(2*x-3)^10",
        "sin(x)^5*cos(x)",
        "ln(x)",
        "x*cos(x)",
        "x^2*e^(1-x)",
        "sin(x)^5",
        "sin(x)^4*cos(x)^3",
        "e^x*e^e^x*e^e^e^x*e^e^e^e^x*e^e^e^e^e^x*e^e^e^e^e^e^x",
        "e^x*(x+1)*ln(x)",
    };

    // add huge sum
    // integrals.push_back(fmt::format("{}", fmt::join(integrals, "+")));

    Performance::test_performance(integrals);

    // const auto integral = Sym::integral(Parser::parse_function(integrals.back()));

    // fmt::print("Trying to solve an integral: {}\n", integral.data()->to_tex());

    // Sym::Integrator integrator;
    // Sym::ComputationHistory history;
    // const auto solution = integrator.solve_integral(integral);//integrator.solve_integral_with_history(integral, history);
    
    // if (solution.has_value()) {
    //     fmt::print("Success! Solution:\n{} + C\n", solution.value().data()->to_tex());

    //     fmt::print("\nComputation steps:\n\n");

    //     for (const auto& step: history.get_tex_history()) {
    //         fmt::print("  {}\\\\\n\n", step);
    //     }
    // }
    // else {
    //     fmt::print("No solution found\n");
    // }
}
