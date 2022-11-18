#include "Solver.cuh"

#include "Evaluation/Integrate.cuh"
#include "Evaluation/StaticFunctions.cuh"

#include "Symbol/Integral.cuh"

Solver::Solver() { Sym::Static::init_functions(); }

std::optional<Expression> Solver::solve(const Expression& integral) const {
    std::vector<Sym::Symbol> integral_input;

    if (integral.symbols[0].is(Sym::Type::Integral)) {
        integral_input = integral.symbols;
    }
    else {
        integral_input = Sym::integral(integral.symbols);
    }

    auto solve_result = Sym::solve_integral(integral_input);

    if (solve_result.has_value()) {
        return Expression(solve_result.value());
    }

    return std::nullopt;
}
