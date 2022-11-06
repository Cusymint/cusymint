#include "ConstantIntegral.cuh"

#include "Symbol/MetaOperators.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_constant_integral(const Integral& integral) {
        const Symbol* const integrand = integral.integrand();
        return integrand->is_constant() ? 1 : 0;
    }

    __device__ void integrate_constant_integral(const Integral& integral, Symbol& destination,
                                                Symbol& /*help_space*/) {
        SolutionOfIntegral<Mul<Var, Copy>>::init(destination, {integral, *integral.integrand()});
    }
}
