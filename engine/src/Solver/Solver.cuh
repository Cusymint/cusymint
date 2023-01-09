#pragma once

#include "../Symbol/Symbol.cuh"
#include "Evaluation/ComputationHistory.cuh"

#include "Expression.cuh"
#include <optional>
#include <vector>

class Solver {
  public:
    Solver();
    [[nodiscard]] std::optional<Expression> solve(const Expression& integral) const;
    [[nodiscard]] std::optional<std::pair<Expression, Sym::ComputationHistory>> solve_with_history(const Expression& integral) const;
};
