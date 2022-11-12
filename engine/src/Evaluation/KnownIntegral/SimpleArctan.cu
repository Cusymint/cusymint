#include "SimpleArctan.cuh"

#include "Symbol/MetaOperators.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_arctan(const Integral& integral) {
        const Symbol* const integrand = integral.integrand();
        // 1/(1+x^2)
        return integrand[0].is(Type::Reciprocal) && integrand[1].is(Type::Addition) &&
                       integrand[2].is(Type::NumericConstant) &&
                       integrand[2].numeric_constant.value == 1.0 && integrand[3].is(Type::Power) &&
                       integrand[6].is(Type::Variable) && integrand[5].is(Type::NumericConstant) &&
                       integrand[5].numeric_constant.value == 2.0
                   ? 1
                   : 0;
    }

    __device__ void integrate_simple_arctan(const Integral& integral, Symbol& destination,
                                            Symbol& /*help_space*/) {
        SolutionOfIntegral<Arctan<Var>>::init(destination, {integral});
    }
}
