#include "PowerFunction.cuh"

#include "Symbol/MetaOperators.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_power_function(const Integral& integral) {
        const Symbol* const integrand = integral.integrand();
        if (!integrand[0].is(Type::Power) || !integrand[1].is(Type::Variable)) {
            return 0;
        }

        if (integrand[2].is(Type::NumericConstant) &&
            integrand[2].as<NumericConstant>().value == -1.0) {
            return 0;
        }

        return integrand[2].is_constant() ? 1 : 0;
    }

    __device__ void integrate_power_function(const Integral& integral, Symbol& destination,
                                             Symbol& /*help_space*/) {
        const Symbol& exponent = integral.integrand()->power.arg2();
        SolutionOfIntegral<Mul<Inv<Add<Copy, Num>>, Pow<Var, Add<Copy, Num>>>>::init(
            destination, {integral, exponent, 1.0, exponent, 1.0});
    }
}
