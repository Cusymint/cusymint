#include "PowerFunction.cuh"

#include "Symbol/MetaOperators.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_power_function(const Integral& integral) {
        return Pow<Var, AllOf<Const, Not<Integer<-1>>>>::match(integral.integrand()) ? 1 : 0;
    }

    __device__ EvaluationStatus integrate_power_function(
        const Integral& integral, const ExpressionArray<>::Iterator& destination,
        const ExpressionArray<>::Iterator& /*help_space*/) {
        const Symbol& exponent = integral.integrand().power.arg2();
        return simple_solution<Mul<Inv<Add<Copy, Num>>, Pow<Var, Add<Copy, Num>>>>(
            destination, {integral, exponent, 1.0, exponent, 1.0});
    }
}
