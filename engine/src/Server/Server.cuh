#pragma once

#include "mongoose.h"
#include <fmt/core.h>
#include <string>
#include <functional>

class Server {
    private:
        struct mg_mgr mgr;
        /// @brief Server URI.
        /// @example "ws://localhost:8000"
        std::string listen_on;

    public:
        Server(std::string listen_on);
        ~Server();
        void run();
};
