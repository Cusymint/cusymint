#include "Solver.cuh"

#include <stdexcept>

#include "Evaluation/Integrator.cuh"
#include "Evaluation/ComputationHistory.cuh"
#include "Evaluation/StaticFunctions.cuh"
#include "Symbol/Integral.cuh"
#include "Symbol/SymbolType.cuh"

Solver::Solver() { Sym::Static::init_functions(); }

void validate_integral(const Expression& integral) {
    if (!integral.symbols[0].is(Sym::Type::Integral)) {
        throw std::invalid_argument(
            fmt::format("Invalid type ({}) of Expression passed to Solver (should be Integral)",
                        Sym::type_name(integral.symbols[0].type())));
    }
}

std::optional<Expression> Solver::solve(const Expression& integral) const {
    validate_integral(integral);

    Sym::Integrator integrator;
    auto solve_result = integrator.solve_integral(integral.symbols);

    if (solve_result.has_value()) {
        return Expression::with_added_constant(solve_result.value());
    }

    return std::nullopt;
}

std::optional<std::pair<Expression, Sym::ComputationHistory>> Solver::solve_with_history(const Expression& integral) const {
    validate_integral(integral);

    Sym::Integrator integrator;
    Sym::ComputationHistory history;
    auto solve_result = integrator.solve_integral_with_history(integral.symbols, history);

    if (solve_result.has_value()) {
        return std::make_pair(
            Expression::with_added_constant(solve_result.value()),
            history
        );
    }

    return std::nullopt;
}
