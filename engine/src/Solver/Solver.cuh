#pragma once

#include <vector>
#include "../Symbol/Symbol.cuh"
#include "Expression.cuh"

class Solver {
    public:
        Solver();
        ~Solver();
        void solve(Expression integral);
}
