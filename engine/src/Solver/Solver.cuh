#pragma once

#include "../Symbol/Symbol.cuh"
#include "Expression.cuh"
#include <optional>
#include <vector>

class Solver {
  public:
    Solver();
    std::optional<Expression> solve(const Expression& integral) const;
};
