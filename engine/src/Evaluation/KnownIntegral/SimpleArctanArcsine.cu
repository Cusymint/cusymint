#include "SimpleArctanArcsine.cuh"

#include "Symbol/MetaOperators.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_arctan(const Integral& integral) {
        return Inv<Add<Integer<1>, Pow<Var, Integer<2>>>>::match(integral.integrand()) ? 1 : 0;
    }

    __device__ EvaluationStatus
    integrate_simple_arctan(const Integral& integral, SymbolIterator& destination,
                            const ExpressionArray<>::Iterator& /*help_space*/) {
        return simple_solution<Arctan<Var>>(destination, {integral});
    }

    __device__ size_t is_simple_arcsine(const Integral& integral) {
        if (!Pow<Sub<Integer<1>, Pow<Var, Integer<2>>>, Num>::match(integral.integrand())) {
            return 0;
        }

        return integral.integrand().as<Power>().arg2().as<NumericConstant>().value == -0.5 ? 1 : 0;
    }

    __device__ EvaluationStatus
    integrate_simple_arcsine(const Integral& integral, SymbolIterator& destination,
                             const ExpressionArray<>::Iterator& /*help_space*/) {
        return simple_solution<Arcsin<Var>>(destination, {integral});
    }
}
