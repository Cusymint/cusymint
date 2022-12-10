#include "SimpleTangentCotangent.cuh"

#include "Symbol/MetaOperators.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_tangent(const Integral& integral) {
        return Inv<Pow<Cos<Var>, Integer<2>>>::match(*integral.integrand()) ? 1 : 0;
    }

    __device__ void integrate_simple_tangent(const Integral& integral, Symbol& destination,
                                             Symbol& /*help_space*/) {
        SolutionOfIntegral<Tan<Var>>::init(destination, {integral});
    }

    __device__ size_t is_simple_cotangent(const Integral& integral) {
        return Inv<Pow<Sin<Var>, Integer<2>>>::match(*integral.integrand()) ? 1 : 0;
    }
    __device__ void integrate_simple_cotangent(const Integral& integral, Symbol& destination,
                                               Symbol& /*help_space*/) {
        SolutionOfIntegral<Neg<Cot<Var>>>::init(destination, {integral});
    }
}