#include "Expression.cuh"
#include "Symbol/Constants.cuh"
#include "Symbol/Integral.cuh"

Expression::Expression(const std::vector<Sym::Symbol>& symbols) : symbols(symbols){};

std::string Expression::to_string() const { return symbols.data()->to_string(); }

std::string Expression::to_tex() const { return symbols.data()->to_tex(); }

Expression Expression::with_added_constant(const std::vector<Sym::Symbol>& symbols) {
    if (symbols.empty() || symbols[0].is(0)) {
        return {Sym::cnst("C")};
    }
    return {symbols + Sym::cnst("C")};
}

Expression Expression::wrap_with_integral(const std::vector<Sym::Symbol>& symbols) {
    if (!symbols[0].is(Sym::Type::Integral)) {
        return {Sym::integral(symbols)};
    }
    return {symbols};
}
