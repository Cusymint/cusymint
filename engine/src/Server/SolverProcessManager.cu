#include "SolverProcessManager.cuh"

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>

#include <fmt/core.h>

std::string exec(const std::string cmd) {
    std::array<char, 4096> buffer{};
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

std::string SolverProcessManager::create_command(const std::string& input) const {
    return fmt::format("./{} --solve-json \"{}\"", executable_name, input);
}

std::string SolverProcessManager::try_solve(const std::string& input) const {
    try {
        auto result = exec(create_command(input));
        if(result.back() == '\n') {
            result.pop_back();
        }
        if(result[0] != '{' || result.back() != '}') {
            throw std::runtime_error("Solver returned invalid JSON");
        }
        return result;
    } catch (const std::exception& e) {
        return R"({"errors": ["Internal runtime error."]})";
    }
}
