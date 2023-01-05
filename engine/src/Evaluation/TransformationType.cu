#include "TransformationType.cuh"

namespace Sym {
    namespace {
        // Space assumed to be not exceeded by the derivative of substitution or integration by
        // parts (improvement possible)
        constexpr size_t DERIVATIVE_SIZE = 1024;
    }

    std::optional<std::shared_ptr<TransformationType>>
    get_transformation_type(const SubexpressionCandidate& candidate,
                            const Integral& previous_integral) {
        if (candidate.arg().is(Type::Integral)) {
            const auto& integral = candidate.arg().as<Integral>();
            if (integral.substitution_count > previous_integral.substitution_count) {
                // substitution happened
                const auto& last_sub = integral.last_substitution().expression();
                std::vector<Symbol> substitution(last_sub.size());
                std::vector<Symbol> derivative(DERIVATIVE_SIZE);
                auto iterator =
                    SymbolIterator::from_at(*derivative.data(), 0, derivative.size()).good();
                last_sub.copy_to(*substitution.data());
                last_sub.derivative_to(iterator).unwrap();
                derivative.resize(derivative.data()->size());

                return std::make_shared<Substitute>(substitution, derivative,
                                                    integral.substitution_count);
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