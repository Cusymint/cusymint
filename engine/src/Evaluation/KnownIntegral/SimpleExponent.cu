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

    __device__ size_t is_power_with_constant_base(const Integral& integral) {
        return Pow<AllOf<Not<E>, Const>, Var>::match(integral.integrand()) ? 1 : 0;
    }
    __device__ EvaluationStatus integrate_power_with_constant_base(const Integral& integral, SymbolIterator& destination,
                             const ExpressionArray<>::Iterator& /*help_space*/) {
        const auto& constant = integral.integrand().as<Power>().arg1();
        return simple_solution<Frac<Pow<Copy, Var>, Ln<Copy>>>(destination,
                                                                 {integral, constant, constant});
    }
}
