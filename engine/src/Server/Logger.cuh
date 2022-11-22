#pragma once

#include <fmt/core.h>
#include "Utils/CompileConstants.cuh"

class Logger {
public:
    static bool is_enabled;

    template <typename... T> static void print(fmt::format_string<T...> fmt, T&&... args) {
        if (is_enabled) {
            fmt::print(fmt, std::forward<T>(args)...);
        }
    }
};
