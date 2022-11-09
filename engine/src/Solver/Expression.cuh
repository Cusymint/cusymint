#pragma once

#include <string>
#include <vector>

#include "../Symbol/Symbol.cuh"

class Expression {
  public:
    Expression(const std::vector<Sym::Symbol>& symbols);

    std::vector<Sym::Symbol> symbols;
    std::string to_string() const;
    std::string to_tex() const;
};
