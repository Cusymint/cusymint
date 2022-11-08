#include "SimpleSineCosine.cuh"

#include "Evaluation/StaticFunctions.cuh"

#include "Symbol/MetaOperators.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_sine(const Integral& integral) {
        const Symbol* const integrand = integral.integrand();
        return integrand[0].is(Type::Sine) && integrand[1].is(Type::Variable) ? 1 : 0;
    }

    __device__ void integrate_simple_sine(const Integral& integral, Symbol& destination,
                                          Symbol& /*help_space*/) {
        SolutionOfIntegral<Neg<Copy>>::init(destination, {integral, Static::cos_x()});
    }

    __device__ size_t is_simple_cosine(const Integral& integral) {
        const Symbol* const integrand = integral.integrand();
        return integrand[0].is(Type::Cosine) && integrand[1].is(Type::Variable) ? 1 : 0;
    }

    __device__ void integrate_simple_cosine(const Integral& integral, Symbol& destination,
                                            Symbol& /*help_space*/) {
        SolutionOfIntegral<Copy>::init(destination, {integral, Static::sin_x()});
    }

}
