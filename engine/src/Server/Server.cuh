#pragma once

#include "mongoose.h"
#include <fmt/core.h>
#include <string>
#include <functional>

#include "../Solver/CachedParser.cuh"

class Server {
    private:
        struct mg_mgr _mgr;
        /// @brief Server URI.
        /// @example "ws://localhost:8000"
        std::string _listen_on;
        CachedParser _cached_parser;

    public:
        Server(std::string listen_on, CachedParser cached_parser);
        ~Server();
        void run();
};
