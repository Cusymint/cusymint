#include "SimpleVariable.cuh"

#include "Symbol/MetaOperators.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_variable(const Integral& integral) {
        return integral.integrand().is(Type::Variable) ? 1 : 0;
    }

    __device__ EvaluationStatus integrate_simple_variable(
        const Integral& integral, const ExpressionArray<>::Iterator& destination,
        const ExpressionArray<>::Iterator& /*help_space*/) {
        return simple_solution<Prod<Num, Pow<Var, Integer<2>>>>(destination, {integral, 0.5});
    }
}
