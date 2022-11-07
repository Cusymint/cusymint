#pragma once

#include <string>

#include "../Solver/Expression.cuh"

class JsonFormatter {
    public:
        std::string format(Expression* input, Expression* output) const;
};
