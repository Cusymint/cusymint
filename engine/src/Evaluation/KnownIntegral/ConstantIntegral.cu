#include "ConstantIntegral.cuh"

#include "Symbol/MetaOperators.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_constant_integral(const Integral& integral) {
        return integral.integrand().is_almost_constant() ? 1 : 0;
    }

    __device__ EvaluationStatus
    integrate_constant_integral(const Integral& integral, SymbolIterator& destination,
                                const ExpressionArray<>::Iterator& /*help_space*/) {
        return simple_solution<Mul<Copy, Var>>(destination, {integral, integral.integrand()});
    }
}
