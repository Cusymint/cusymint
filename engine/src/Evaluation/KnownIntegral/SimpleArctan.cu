#include "SimpleArctan.cuh"

#include "Symbol/MetaOperators.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_arctan(const Integral& integral) {
        return Inv<Add<Integer<1>, Pow<Var, Integer<2>>>>::match(*integral.integrand()) ? 1 : 0;
    }

    __device__ void integrate_simple_arctan(const Integral& integral, Symbol& destination,
                                            Symbol& /*help_space*/) {
        SolutionOfIntegral<Arctan<Var>>::init(destination, {integral});
    }
}
