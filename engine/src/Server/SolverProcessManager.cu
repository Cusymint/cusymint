#include "SolverProcessManager.cuh"

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>

#include <fmt/core.h>

std::string exec(const std::string cmd) {
    std::array<char, 4096> buffer;
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

std::string create_command(std::string input) {
    std::string executable_name = "cusymint";

    std::string command = fmt::format("./{} --solve-json {}", executable_name, input);
    
    return command;
}

std::string get_result(std::string input) {
    try {
        std::string command = create_command(input);
        std::string result = exec(command);
        return result;
    } catch (const std::exception& e) {
        return R"({"errors": ["Internal runtime error."]})";
    }
}
