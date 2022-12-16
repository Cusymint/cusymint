#include "SimpleExponent.cuh"

#include "Evaluation/StaticFunctions.cuh"

#include "Symbol/MetaOperators.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_exponent(const Integral& integral) {
        return Pow<E, Var>::match(integral.integrand()) ? 1 : 0;
    }

    __device__ EvaluationStatus integrate_simple_exponent(
        const Integral& integral, SymbolIterator& destination,
        const ExpressionArray<>::Iterator& /*help_space*/) {
        return simple_solution<Pow<E, Var>>(destination, {integral});
    }
}
