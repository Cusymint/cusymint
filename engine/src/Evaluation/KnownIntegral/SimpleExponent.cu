#include "SimpleExponent.cuh"

#include "Evaluation/StaticFunctions.cuh"

#include "Symbol/MetaOperators.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_exponent(const Integral& integral) {
        return Pow<E, Var>::match(*integral.integrand()) ? 1 : 0;
    }

    __device__ void integrate_simple_exponent(const Integral& integral, Symbol& destination,
                                              Symbol& /*help_space*/) {
        SolutionOfIntegral<Copy>::init(destination, {integral, Static::e_to_x()});
    }
}
