#include <fmt/core.h>

#include "Evaluation/StaticFunctions.cuh"
#include "Solver/CachedParser.cuh"
#include "Solver/Solver.cuh"
#include "Server/Server.cuh"

#include "Utils/CompileConstants.cuh"

int main() {
    if constexpr (Consts::DEBUG) {
        fmt::print("Running in debug mode\n");
    }
    
    Sym::Static::init_functions();

    auto uri = "ws://localhost:8000";
    CachedParser parser;
    Solver solver;
    Server server = Server(uri, parser, solver);
    server.run();
}