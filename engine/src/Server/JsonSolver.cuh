#pragma once

#include <string>
#include <utility>

#include "../Solver/Solver.cuh"
#include "../Solver/CachedParser.cuh"

class JsonSolver {
    private:
        Solver solver;
        CachedParser parser;
    
    public:
        explicit JsonSolver(const Solver& solver, CachedParser parser) : solver(solver), parser(std::move(parser)) {};
        std::string try_solve(const std::string& input);
};