#pragma once

#include <string>

#include "../Solver/Expression.cuh"

class JsonFormatter {
  public:
    std::string format(Expression* input, Expression* output, std::vector<std::string>* errors) const;
};
