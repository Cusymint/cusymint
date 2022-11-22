#pragma once

#include <fmt/core.h>
#include "Utils/CompileConstants.cuh"

class Logger {
public:
    static bool isEnabled;

    template <typename... T> static void print(fmt::format_string<T...> fmt, T&&... args) {
        if (isEnabled) {
            fmt::print(fmt, std::forward<T>(args)...);
        }
    }
};
