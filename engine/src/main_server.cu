#include <fmt/core.h>

#include "Evaluation/StaticFunctions.cuh"
#include "Server/Server.cuh"
#include "Solver/CachedParser.cuh"
#include "Solver/Solver.cuh"

#include "Utils/CompileConstants.cuh"

int main(int argc, char** argv) {
    if constexpr (Consts::DEBUG) {
        fmt::print("Running in debug mode\n");
    }

    const auto* default_uri = "ws://localhost:8000";
    const char* uri = argc > 1 ? argv[1] : default_uri;

    CachedParser parser;
    Solver solver;
    Server server = Server(uri, parser, solver);

    server.run();
}