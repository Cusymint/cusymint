#pragma once

#include <fmt/core.h>
#include <functional>
#include <mongoose.h>
#include <string>

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
