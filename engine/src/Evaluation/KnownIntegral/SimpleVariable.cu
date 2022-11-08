#include "SimpleVariable.cuh"

#include "Symbol/MetaOperators.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_variable(const Integral& integral) {
        return integral.integrand()->is(Type::Variable) ? 1 : 0;
    }

    __device__ void integrate_simple_variable(const Integral& integral, Symbol& destination,
                                              Symbol& /*help_space*/) {
        SolutionOfIntegral<Prod<Num, Pow<Var, Int<2>>>>::init(destination, {integral, 0.5});
    }
}
