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

#include "Server/Server.cuh"
#include "Solver/CachedParser.cuh"
#include "Solver/Solver.cuh"

#include "Parser/Parser.cuh"

#include "Utils/CompileConstants.cuh"

void run_server() {
    auto uri = "ws://localhost:8000";
    CachedParser parser;
    Solver solver;
    Server server = Server(uri, parser, solver);
    server.run();
}

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

    run_server();

    Sym::Static::init_functions();

    const auto integral = Sym::integral(parse_function(e_tower(11)));

    fmt::print("Trying to solve an integral: {}\n", integral.data()->to_tex());

    const auto solution = Sym::solve_integral(integral);

    if (solution.has_value()) {
        fmt::print("Success! Solution:\n{} + C\n", solution.value().data()->to_tex());
    }
    else {
        fmt::print("No solution found\n");
    }
}
