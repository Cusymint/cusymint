#pragma once

#include <vector>
#include "../Symbol/Symbol.cuh"

class Solver {
    public:
        Solver();
        ~Solver();
        void solve(std::vector<std::Symbol> integral);
}