#include <cstdlib>
#include <cstring>

#include <iostream>
#include <optional>
#include <vector>

#include <fmt/core.h>

#include "Evaluation/Integrate.cuh"
#include "Evaluation/StaticFunctions.cuh"

#include "Symbol/Integral.cuh"
#include "Symbol/Symbol.cuh"

#include "Parser/Parser.cuh"

#include "Utils/CompileConstants.cuh"

int main() {
    if constexpr (Consts::DEBUG) {
        fmt::print("Running in debug mode\n");
    }

    Sym::Static::init_functions();

    auto integral = Sym::integral(parse_function("x^2+x^4+e^(ln(x)*5)+2^(x/ln(2))*e^e^x"));

    fmt::print("Trying to solve an integral: {}\n", integral.data()->to_tex());

    std::optional<std::vector<std::vector<Sym::Symbol>>> solution = Sym::solve_integral(integral);

    if (solution.has_value()) {
        fmt::print("Success! Expressions tree:\n");
        for (const auto& expr : solution.value()) {
            fmt::print("{}\n", expr.data()->to_string());
        }
    }
    else {
        fmt::print("No solution found\n");
    }
}
