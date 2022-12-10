#include "NonElementary.cuh"

#include "Symbol/MetaOperators.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_erf(const Integral& integral) {
        return Pow<E, Neg<Pow<Var, Integer<2>>>>::match(*integral.integrand()) ? 1 : 0;
    }

    __device__ void integrate_simple_erf(const Integral& integral, Symbol& destination,
                                         Symbol& /*help_space*/) {
        SolutionOfIntegral<Prod<Sqrt<Pi>, Inv<Integer<2>>, Erf<Var>>>::init(destination,
                                                                            {integral});
    }

    __device__ size_t is_simple_Si(const Integral& integral) {
        return AnyOf<Mul<Inv<Var>, Sin<Var>>, Frac<Sin<Var>, Var>>::match(*integral.integrand()) ? 1 : 0;
    }

    __device__ void integrate_simple_Si(const Integral& integral, Symbol& destination,
                                        Symbol& /*help_space*/) {
        SolutionOfIntegral<Si<Var>>::init(destination, {integral});
    }

    __device__ size_t is_simple_Ci(const Integral& integral) {
        return AnyOf<Mul<Inv<Var>, Cos<Var>>, Frac<Cos<Var>, Var>>::match(*integral.integrand()) ? 1 : 0;
    }

    __device__ void integrate_simple_Ci(const Integral& integral, Symbol& destination,
                                        Symbol& /*help_space*/) {
        SolutionOfIntegral<Ci<Var>>::init(destination, {integral});
    }

    __device__ size_t is_simple_Ei(const Integral& integral) {
        return AnyOf<Mul<Inv<Var>, Pow<E, Var>>, Frac<Pow<E, Var>, Var>>::match(*integral.integrand()) ? 1 : 0;
    }

    __device__ void integrate_simple_Ei(const Integral& integral, Symbol& destination,
                                        Symbol& /*help_space*/) {
        SolutionOfIntegral<Ei<Var>>::init(destination, {integral});
    }

    __device__ size_t is_simple_li(const Integral& integral) {
        return Inv<Ln<Var>>::match(*integral.integrand()) ? 1 : 0;
    }

    __device__ void integrate_simple_li(const Integral& integral, Symbol& destination,
                                        Symbol& /*help_space*/) {
        SolutionOfIntegral<Li<Var>>::init(destination, {integral});
    }
}