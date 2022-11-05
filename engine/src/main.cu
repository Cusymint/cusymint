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

    const auto integral = Sym::integral(parse_function("sin(x)+cos(x)+e^e^x*e^x+e^x*sin(e^x)"));

    fmt::print("Trying to solve an integral: {}\n", integral.data()->to_tex());

    const auto solution = Sym::solve_integral(integral);

    if (solution.has_value()) {
        fmt::print("Success! Solution:\n{}", solution.value().data()->to_tex());
    }
    else {
        fmt::print("No solution found\n");
    }
}
