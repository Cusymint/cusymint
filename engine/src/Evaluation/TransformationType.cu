#include "Symbol/SubexpressionCandidate.cuh"
#include "Symbol/SymbolType.cuh"
#include "TransformationType.cuh"

#include <memory>
#include <vector>

#include "Evaluation/Integrator.cuh"

namespace Sym {
    namespace {
        // Space assumed to be not exceeded by the derivative of substitution or integration by
        // parts (improvement possible)
        constexpr size_t DERIVATIVE_SIZE = 1024;
    }

    std::optional<std::shared_ptr<TransformationType>>
    get_transformation_type(const SubexpressionCandidate& candidate,
                            const Integral& previous_integral,
                            const std::vector<std::vector<Symbol>> expression_tree) {
        if (candidate.arg().is(Type::Integral)) {
            const auto& integral = candidate.arg().as<Integral>();

            if (integral.substitution_count > previous_integral.substitution_count) {
                // substitution happened
                const auto& last_sub = integral.last_substitution().expression();
                std::vector<Symbol> substitution(last_sub.size());
                std::vector<Symbol> derivative(DERIVATIVE_SIZE);
                std::vector<Symbol> help_space(DERIVATIVE_SIZE * Integrator::HELP_SPACE_MULTIPLIER);

                auto derivative_iterator =
                    SymbolIterator::from_at(*derivative.data(), 0, derivative.size()).good();
                auto help_space_iterator =
                    SymbolIterator::from_at(*help_space.data(), 0, help_space.size()).good();

                last_sub.copy_to(*substitution.data());
                last_sub.derivative_to(derivative_iterator).unwrap();

                derivative.data()->simplify(help_space_iterator).unwrap();
                derivative.resize(derivative.data()->size());

                return std::make_shared<Substitute>(substitution, derivative,
                                                    integral.substitution_count);
            }
        }

        if (candidate.arg().is(Type::Addition)) {
            const auto& addition = candidate.arg().as<Addition>();
            if (addition.arg1().is(Type::Product) && addition.arg1().as<Product>().arg1().is(-1)) {
                // integration by parts happened
                const auto& vacancy =
                    addition.arg1().as<Product>().arg2().as<SubexpressionVacancy>();
                const auto& child_integral =
                    expression_tree[vacancy.solver_idx][0].as<SubexpressionCandidate>().arg().as<Integral>();
                const auto& first_function = addition.arg2().as<Product>().arg1();
                const auto& second_function = addition.arg2().as<Product>().arg2();
                const auto& second_function_derivative =
                    child_integral.integrand().as<Product>().arg2();

                std::vector<Symbol> first(first_function.size());
                std::vector<Symbol> second(second_function.size());
                std::vector<Symbol> second_derivative(second_function_derivative.size());
                std::vector<Symbol> first_derivative(DERIVATIVE_SIZE);
                std::vector<Symbol> help_space(DERIVATIVE_SIZE * Integrator::HELP_SPACE_MULTIPLIER);

                auto derivative_iterator =
                    SymbolIterator::from_at(*first_derivative.data(), 0, first_derivative.size())
                        .good();
                auto help_space_iterator =
                    SymbolIterator::from_at(*help_space.data(), 0, help_space.size()).good();

                first_function.copy_to(*first.data());
                second_function.copy_to(*second.data());
                second_function_derivative.copy_to(*second_derivative.data());

                first_function.derivative_to(derivative_iterator).unwrap();

                first_derivative.data()->simplify(help_space_iterator).unwrap();
                first_derivative.resize(first_derivative.data()->size());

                second_derivative.data()->simplify(help_space_iterator).unwrap();
                second_derivative.resize(second_derivative.data()->size());

                return std::make_shared<IntegrateByParts>(first, second, first_derivative,
                                                          second_derivative,
                                                          child_integral.substitution_count);
            }
        }

        if (candidate.arg().is(Type::Solution)) {
            // solution happened
            const auto& solution_arg = candidate.arg().as<Solution>().expression();
            std::vector<Symbol> integrand(previous_integral.integrand().size());
            std::vector<Symbol> solution(solution_arg.size());

            solution_arg.copy_to(*solution.data());
            previous_integral.integrand().copy_to(*integrand.data());

            return std::make_shared<SolveIntegral>(integral(integrand), solution,
                                                   previous_integral.substitution_count);
        }

        return std::nullopt;
    }

}