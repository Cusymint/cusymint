#include "Expression.cuh"

Expression::Expression(const std::vector<Sym::Symbol>& symbols) : symbols(symbols){};

std::string Expression::to_string() const { return symbols.data()->to_string(); }

std::string Expression::to_tex() const { return symbols.data()->to_tex(); }
