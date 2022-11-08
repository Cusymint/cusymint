#pragma once

#include <mongoose.h>
#include <fmt/core.h>
#include <string>
#include <functional>

#include "../Solver/CachedParser.cuh"
#include "../Solver/Solver.cuh"

class Server {
    private:
        struct mg_mgr _mgr;
        /// @brief Server URI.
        /// @example "ws://localhost:8000"
        std::string _listen_on;
        CachedParser _cached_parser;
        Solver _solver;

    public:
        Server(std::string listen_on, CachedParser cached_parser, Solver solver);
        ~Server();
        void run();
};
