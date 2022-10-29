#include "SimpleExponent.cuh"

#include "Evaluation/StaticFunctions.cuh"

#include "Symbol/MetaOperators.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_exponent(const Integral& integral) {
        const Symbol* const integrand = integral.integrand();
        return integrand[0].is(Type::Power) && integrand[1].is(Type::KnownConstant) &&
                       integrand[1].as<KnownConstant>().value == KnownConstantValue::E &&
                       integrand[2].is(Type::Variable)
                   ? 1
                   : 0;
    }

    __device__ void integrate_simple_exponent(const Integral& integral, Symbol& destination,
                                              Symbol& /*help_space*/) {
        SolutionOfIntegral<Copy>::init(destination, {integral, Static::e_to_x()});
    }
}
