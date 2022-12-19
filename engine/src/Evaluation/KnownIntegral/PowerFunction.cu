#include "PowerFunction.cuh"

#include "Symbol/MetaOperators.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_power_function(const Integral& integral) {
        return Pow<Var, AllOf<Const, Not<Integer<-1>>>>::match(integral.integrand()) ? 1 : 0;
    }

    __device__ EvaluationStatus integrate_power_function(
        const Integral& integral, SymbolIterator& destination,
        const ExpressionArray<>::Iterator& /*help_space*/) {
        const Symbol& exponent = integral.integrand().as<Power>().arg2();
        return simple_solution<Mul<Inv<Add<Copy, Num>>, Pow<Var, Add<Copy, Num>>>>(
            destination, {integral, exponent, 1.0, exponent, 1.0});
    }

    __device__ size_t is_reciprocal(const Integral& integral) {
        return AnyOf<Inv<Var>, Pow<Var, Integer<-1>>>::match(integral.integrand()) ? 1 : 0;
    }

    __device__ EvaluationStatus integrate_reciprocal(const Integral& integral, SymbolIterator& destination,
        const ExpressionArray<>::Iterator& /*help_space*/) {
        return simple_solution<Ln<Var>>(destination, {integral});
    }
}
