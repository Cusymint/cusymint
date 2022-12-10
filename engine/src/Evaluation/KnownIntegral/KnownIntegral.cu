#include "KnownIntegral.cuh"

#include "Utils/Meta.cuh"

#include "ConstantIntegral.cuh"
#include "NonElementary.cuh"
#include "PowerFunction.cuh"
#include "SimpleArctanArcsine.cuh"
#include "SimpleExponent.cuh"
#include "SimpleSineCosine.cuh"
#include "SimpleTangentCotangent.cuh"
#include "SimpleVariable.cuh"

namespace Sym::KnownIntegral {
    __device__ const Check CHECKS[] = {
        is_simple_variable, is_power_function,    is_simple_exponent,  is_simple_sine,
        is_simple_cosine,   is_constant_integral, is_simple_arctan,    is_simple_arcsine,
        is_reciprocal,      is_simple_tangent,    is_simple_cotangent, is_power_with_constant_base,
        is_simple_erf,      is_simple_Si,         is_simple_Ci,        is_simple_Ei,
        is_simple_li,
    };

    __device__ const Application APPLICATIONS[] = {
        integrate_simple_variable, integrate_power_function,   integrate_simple_exponent,
        integrate_simple_sine,     integrate_simple_cosine,    integrate_constant_integral,
        integrate_simple_arctan,   integrate_simple_arcsine,   integrate_reciprocal,
        integrate_simple_tangent,  integrate_simple_cotangent, integrate_power_with_constant_base,
        integrate_simple_erf,      integrate_simple_Si,        integrate_simple_Ci,
        integrate_simple_Ei,       integrate_simple_li,
    };

#ifdef __CUDA_ARCH__
    __device__
#endif
        const size_t COUNT =
            Util::ensure_same_v<Util::array_len(CHECKS), Util::array_len(APPLICATIONS)>;
}
