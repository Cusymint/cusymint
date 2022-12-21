#include "Evaluation/KnownIntegral/KnownIntegral.cuh"
#include "SimpleTangentCotangent.cuh"

#include "Symbol/MetaOperators.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_tangent(const Integral& integral) {
        return Inv<Pow<Cos<Var>, Integer<2>>>::match(integral.integrand()) ? 1 : 0;
    }

    __device__ EvaluationStatus integrate_simple_tangent(const Integral& integral, SymbolIterator& destination,
                             const ExpressionArray<>::Iterator& /*help_space*/) {
        return simple_solution<Tan<Var>>(destination, {integral});
    }

    __device__ size_t is_simple_cotangent(const Integral& integral) {
        return Inv<Pow<Sin<Var>, Integer<2>>>::match(integral.integrand()) ? 1 : 0;
    }
    __device__ EvaluationStatus integrate_simple_cotangent(const Integral& integral, SymbolIterator& destination,
                             const ExpressionArray<>::Iterator& /*help_space*/) {
        return simple_solution<Neg<Cot<Var>>>(destination, {integral});
    }
}