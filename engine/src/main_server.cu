#include <fmt/core.h>

#include "Evaluation/StaticFunctions.cuh"
#include "Server/Server.cuh"
#include "Server/Logger.cuh"
#include "Server/JsonSolver.cuh"
#include "Solver/CachedParser.cuh"
#include "Solver/Solver.cuh"

#include "Utils/CompileConstants.cuh"

int main(int argc, char** argv) {
    const auto* default_uri = "ws://localhost:8000";
    CachedParser parser;
    Solver solver;

    if (argc == 3 && std::string(argv[1]) == "--solve-json") {
        std::string input = argv[2];

        auto json_solver = JsonSolver(solver, parser);

        fmt::print("{}\n", json_solver.try_solve(input));
        return EXIT_SUCCESS;
    }

    if (argc == 3 && std::string(argv[1]) == "--solve-with-steps-json") {
        std::string input = argv[2];

        auto json_solver = JsonSolver(solver, parser);

        fmt::print("{}\n", json_solver.try_solve_with_steps(input));
        return EXIT_SUCCESS;
    }

    if constexpr (Consts::DEBUG) {
        fmt::print("Running in debug mode\n");
    }

    const char* uri = argc > 1 ? argv[1] : default_uri;

    Logger::is_enabled = true;
    Server server = Server(uri, parser);

    server.run();
}