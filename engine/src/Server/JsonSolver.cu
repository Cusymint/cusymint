#include "JsonSolver.cuh"
#include "ResponseBuilder.cuh"

std::string JsonSolver::try_solve(const std::string& input) {
    auto response_builder = ResponseBuilder();

    try {
        auto parser_result = Expression::wrap_with_integral(parser.parse(input).symbols);
        response_builder.set_input(parser_result);
        auto solver_result = solver.solve(parser_result);
        if (solver_result.has_value()) {
            response_builder.set_output(solver_result.value());
        }
        else {
            response_builder.add_error("No solution found");
        }
    } catch (const std::invalid_argument& e) {
        response_builder.add_error(e.what());
    } catch (const std::exception& e) {
        response_builder.add_error("Internal error");
    }

    return response_builder.get_json_response();
}
