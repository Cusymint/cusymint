#include "Solver.cuh"

#include "../Evaluation/Integrate.cuh"
#include "../Evaluation/StaticFunctions.cuh"

Solver::Solver() { Sym::Static::init_functions(); }

std::optional<Expression> Solver::solve(const Expression& integral) const {
    auto solve_result = Sym::solve_integral(integral.symbols);

    if (solve_result.has_value()) {
        return Expression(solve_result.value());
    }

    return std::nullopt;
}
