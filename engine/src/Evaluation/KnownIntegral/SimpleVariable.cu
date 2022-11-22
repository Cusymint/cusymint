#include "SimpleVariable.cuh"

#include "Symbol/MetaOperators.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_variable(const Integral& integral) {
        return integral.integrand()->is(Type::Variable) ? 1 : 0;
    }

    __device__ EvaluationStatus integrate_simple_variable(const Integral& integral,
                                                          Symbol& destination,
                                                          Symbol& /*help_space*/) {
        // TODO: ehhhh trzeba to policzyć lub napisać coś co to policzy ENSURE_ENOUGH_SPACE(/**/,
        // destination);
        SolutionOfIntegral<Prod<Num, Pow<Var, Integer<2>>>>::init(destination, {integral, 0.5});
        /* return EvaluationStatus::Done; */
    }
}
