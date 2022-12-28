#include <cstdlib>
#include <cstring>

#include <iostream>
#include <optional>
#include <vector>

#include <fmt/core.h>

#include "Evaluation/Integrator.cuh"
#include "Evaluation/StaticFunctions.cuh"

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

    const auto integral = Sym::integral(Parser::parse_function("ctg(x)^2"));

    fmt::print("Trying to solve an integral: {}\n", integral.data()->to_tex());

    Sym::Integrator integrator;
    Sym::ComputationHistory history;
    const auto solution = integrator.solve_integral_with_history(integral, history);

    history.print_history();

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
