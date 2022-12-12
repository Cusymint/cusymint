#ifndef TEST_SIMPLIFY_CUH
#define TEST_SIMPLIFY_CUH

#include <vector>

#include "Evaluation/Integrator.cuh"
#include "Symbol/Symbol.cuh"

namespace Test {
    void simplify_vector(std::vector<Sym::Symbol>& expression) {
        while (true) {
            std::vector<Sym::Symbol> expr_copy(expression);
            std::vector<Sym::Symbol> simplification_memory(Sym::Integrator::HELP_SPACE_MULTIPLIER *
                                                           expr_copy.size());

            auto help_space_it = Sym::SymbolIterator::from_at(*simplification_memory.data(), 0,
                                                              simplification_memory.size())
                                     .good();
            const auto result = expr_copy.data()->simplify(help_space_it);

            if (result.is_good()) {
                expr_copy.data()->copy_to(*expression.data());
                break;
            }

            // Sometimes simplified expressions take more space than before, so this is
            // necessary
            expression.resize(Sym::Integrator::REALLOC_MULTIPLIER * expression.size());
        }
        expression.resize(expression.data()->size());
    }
}

#endif