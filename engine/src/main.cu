#include <cstdlib>
#include <cstring>

#include <iostream>
#include <optional>
#include <vector>

#include <fmt/core.h>

#include "Evaluation/Integrate.cuh"
#include "Evaluation/StaticFunctions.cuh"

#include "Symbol/Constants.cuh"
#include "Symbol/ExpressionArray.cuh"
#include "Symbol/Integral.cuh"
#include "Symbol/Symbol.cuh"
#include "Symbol/Variable.cuh"

#include "Parser/Parser.cuh"

#include "Utils/CompileConstants.cuh"

// void print_polynomial_ranks(const Sym::ExpressionArray<Sym::Integral> integrals) {
//     const auto h_integrals = integrals.to_vector();

//     fmt::print("Polynomial ranks:({}):\n", integrals.size());
//     for (size_t int_idx = 0; int_idx < integrals.size(); ++int_idx) {
//         fmt::print("{}: {}\n", int_idx, h_integrals[int_idx].data()->as<Sym::Integral>().integrand()->is_polynomial());
//     }

//     fmt::print("\n");
// }

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

    const auto integral = Sym::integral(Parser::parse_function("(x^4)/(1+x^2)"));

    fmt::print("Trying to solve an integral: {}\n", integral.data()->to_tex());

    const auto solution = Sym::solve_integral(integral);

    if (solution.has_value()) {
        fmt::print("Success! Solution:\n{} + C\n", solution.value().data()->to_tex());
    }
    else {
        fmt::print("No solution found\n");
    }
}
