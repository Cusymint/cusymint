#include "XInverse.cuh"

#include "Symbol/MetaOperators.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_x_inverse(const Integral& integral) {
        return Inv<Var>::match(integral.integrand()) ? 1 : 0;
    }

    __device__ EvaluationStatus
    integrate_x_inverse(const Integral& integral, SymbolIterator& destination,
                        const ExpressionArray<>::Iterator& /*help_space*/) {
        return simple_solution<Ln<Mul<Sgn<Var>, Var>>>(destination, {integral});
    }
}
