#include "ConstantIntegral.cuh"

#include "Symbol/MetaOperators.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_constant_integral(const Integral& integral) {
        const Symbol* const integrand = integral.integrand();
        return integrand->is_constant() ? 1 : 0;
    }

    __device__ EvaluationStatus integrate_constant_integral(const Integral& integral,
                                                            Symbol& destination,
                                                            Symbol& /*help_space*/) {
        // TODO: ehhhh znowu trzeba to policzyć lub napisać itd ENSURE_ENOUGH_SPACE(3 +
        // integral.integrand()->size(), destination);
        SolutionOfIntegral<Mul<Var, Copy>>::init(destination, {integral, *integral.integrand()});
        /* return EvaluationStatus::Done; */
    }
}
