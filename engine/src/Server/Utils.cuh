#pragma once

#include <fmt/core.h>
#include "Utils/CompileConstants.cuh"

template <typename... T> static void print_debug(fmt::format_string<T...> fmt, T&&... args) {
    if constexpr (Consts::DEBUG) {
        fmt::print(fmt, std::forward<T>(args)...);
    }
}
