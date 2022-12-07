#include "Solver.cuh"

#include <stdexcept>

#include "Evaluation/Integrator.cuh"
#include "Evaluation/StaticFunctions.cuh"
#include "Symbol/Integral.cuh"
#include "Symbol/SymbolType.cuh"

Solver::Solver() { Sym::Static::init_functions(); }

std::optional<Expression> Solver::solve(const Expression& integral) const {
    if (!integral.symbols[0].is(Sym::Type::Integral)) {
        throw std::invalid_argument(
            fmt::format("Invalid type ({}) of Expression passed to Solver (should be Integral)",
                        Sym::type_name(integral.symbols[0].type())));
    }

    Sym::Integrator integrator;
    auto solve_result = integrator.solve_integral(integral.symbols);

    if (solve_result.has_value()) {
        return Expression::with_added_constant(solve_result.value());
    }

    return std::nullopt;
}
