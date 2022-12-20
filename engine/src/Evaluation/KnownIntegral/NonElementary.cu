#include "NonElementary.cuh"

#include "Symbol/MetaOperators.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_erf(const Integral& integral) {
        return Pow<E, Neg<Pow<Var, Integer<2>>>>::match(integral.integrand()) ? 1 : 0;
    }

    __device__ EvaluationStatus integrate_simple_erf(const Integral& integral, SymbolIterator& destination,
                             const ExpressionArray<>::Iterator& /*help_space*/) {
        return simple_solution<Prod<Sqrt<Pi>, Inv<Integer<2>>, Erf<Var>>>(destination,
                                                                            {integral});
    }

    __device__ size_t is_simple_Si(const Integral& integral) {
        return AnyOf<Mul<Inv<Var>, Sin<Var>>, Frac<Sin<Var>, Var>>::match(integral.integrand()) ? 1 : 0;
    }

    __device__ EvaluationStatus integrate_simple_Si(const Integral& integral, SymbolIterator& destination,
                             const ExpressionArray<>::Iterator& /*help_space*/) {
        return simple_solution<Si<Var>>(destination, {integral});
    }

    __device__ size_t is_simple_Ci(const Integral& integral) {
        return AnyOf<Mul<Inv<Var>, Cos<Var>>, Frac<Cos<Var>, Var>>::match(integral.integrand()) ? 1 : 0;
    }

    __device__ EvaluationStatus integrate_simple_Ci(const Integral& integral, SymbolIterator& destination,
                             const ExpressionArray<>::Iterator& /*help_space*/) {
        return simple_solution<Ci<Var>>(destination, {integral});
    }

    __device__ size_t is_simple_Ei(const Integral& integral) {
        return AnyOf<Mul<Inv<Var>, Pow<E, Var>>, Frac<Pow<E, Var>, Var>>::match(integral.integrand()) ? 1 : 0;
    }

    __device__ EvaluationStatus integrate_simple_Ei(const Integral& integral, SymbolIterator& destination,
                             const ExpressionArray<>::Iterator& /*help_space*/) {
        return simple_solution<Ei<Var>>(destination, {integral});
    }

    __device__ size_t is_simple_li(const Integral& integral) {
        return Inv<Ln<Var>>::match(integral.integrand()) ? 1 : 0;
    }

    __device__ EvaluationStatus integrate_simple_li(const Integral& integral, SymbolIterator& destination,
                             const ExpressionArray<>::Iterator& /*help_space*/) {
        return simple_solution<Li<Var>>(destination, {integral});
    }
}