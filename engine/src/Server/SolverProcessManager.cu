#include "SolverProcessManager.cuh"
#include "Logger.cuh"

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>

#include <fmt/core.h>

std::string exec_and_read_output(const std::string cmd) {
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
        auto command = create_command(input);
        Logger::print("[SolverProcessManager] Trying to solve with command: {}\n", create_command(input));
        auto result = exec_and_read_output(command);
        if(result.back() == '\n') {
            result.pop_back();
        }
        Logger::print("[SolverProcessManager] Solver returned: {}\n", result);
        if(result[0] != '{' || result.back() != '}') {
            throw std::runtime_error("Solver returned invalid JSON");
        }
        return result;
    } catch (const std::exception& e) {
        Logger::print("[SolverProcessManager] Solver failed: {}\n", e.what());
        return R"({"errors": ["Internal runtime error."]})";
    }
}
