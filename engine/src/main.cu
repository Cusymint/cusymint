#include <cstdlib>
#include <cstring>

#include <iostream>
#include <optional>
#include <vector>

#include <fmt/core.h>

#include "Evaluation/Integrate.cuh"

#include "Symbol/Integral.cuh"
#include "Symbol/Symbol.cuh"

#include "Parser/Parser.cuh"

#include "Utils/CompileConstants.cuh"

int main() {
    if constexpr (Consts::DEBUG) {
        fmt::print("Running in debug mode\n");
    }

    auto integral = Sym::integral(parse_function("x^2+x^4+x^5+log_c(c^e^x)*e^e^x"));
    // std::vector<Sym::Symbol> integral = Sym::integral(
    //     (Sym::var() ^ Sym::num(2)) + (Sym::var() ^ Sym::num(4)) + (Sym::var() ^ Sym::num(5)) +
    //     ((Sym::e() ^ Sym::var()) * (Sym::e() ^ (Sym::e() ^ Sym::var()))));

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
