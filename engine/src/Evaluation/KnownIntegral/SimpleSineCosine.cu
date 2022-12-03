#include "SimpleSineCosine.cuh"

#include "Evaluation/StaticFunctions.cuh"

#include "Symbol/MetaOperators.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_sine(const Integral& integral) {
        const Symbol& integrand = integral.integrand();
        return integrand[0].is(Type::Sine) && integrand[1].is(Type::Variable) ? 1 : 0;
    }

    __device__ EvaluationStatus
    integrate_simple_sine(const Integral& integral, const ExpressionArray<>::Iterator& destination,
                          const ExpressionArray<>::Iterator& /*help_space*/) {
        return simple_solution<Neg<Cos<Var>>>(destination, {integral});
    }

    __device__ size_t is_simple_cosine(const Integral& integral) {
        const Symbol& integrand = integral.integrand();
        return integrand[0].is(Type::Cosine) && integrand[1].is(Type::Variable) ? 1 : 0;
    }

    __device__ EvaluationStatus integrate_simple_cosine(
        const Integral& integral, const ExpressionArray<>::Iterator& destination,
        const ExpressionArray<>::Iterator& /*help_space*/) {
        return simple_solution<Sin<Var>>(destination, {integral});
    }
}
