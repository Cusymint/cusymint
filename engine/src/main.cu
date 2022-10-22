#include <cstdlib>
#include <cstring>

#include <iostream>
#include <optional>
#include <vector>

#include <fmt/core.h>

#include "Evaluation/Integrate.cuh"

#include "Symbol/Constants.cuh"
#include "Symbol/ExpressionArray.cuh"
#include "Symbol/Integral.cuh"
#include "Symbol/Symbol.cuh"
#include "Symbol/Variable.cuh"

#include "Utils/CompileConstants.cuh"

void print_polynomial_ranks(const Sym::ExpressionArray<Sym::Integral> integrals) {
    const auto h_integrals = integrals.to_vector();

    fmt::print("Polynomial ranks:({}):\n", integrals.size());
    for (size_t int_idx = 0; int_idx < integrals.size(); ++int_idx) {
        fmt::print("{}: {}\n", int_idx, h_integrals[int_idx].data()->as<Sym::Integral>().integrand()->is_polynomial());
    }

    fmt::print("\n");
}

int main() {
    if constexpr (Consts::DEBUG) {
        fmt::print("Running in debug mode\n");
    }

    std::vector<Sym::Symbol> integral = Sym::integral(
        (Sym::var() ^ Sym::num(2)) + (Sym::var() ^ Sym::num(4)) + (Sym::var() ^ Sym::num(5)) +
        ((Sym::e() ^ Sym::var()) * (Sym::e() ^ (Sym::e() ^ Sym::var()))));

    fmt::print("Trying to solve an integral: {}\n", integral.data()->to_string());

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
