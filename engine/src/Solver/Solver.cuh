#pragma once

#include <vector>
#include <optional>
#include "../Symbol/Symbol.cuh"
#include "Expression.cuh"

class Solver {
    public:
        Solver();
        std::optional<Expression> solve(const Expression& integral) const;
};
