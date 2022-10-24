#include "KnownIntegral.cuh"

#include "Utils/Meta.cuh"

#include "ConstantIntegral.cuh"
#include "PowerFunction.cuh"
#include "SimpleArctan.cuh"
#include "SimpleExponent.cuh"
#include "SimpleSineCosine.cuh"
#include "SimpleVariable.cuh"

namespace Sym::KnownIntegral {
    __device__ const Check CHECKS[] = {
        is_simple_variable, is_power_function,    is_simple_exponent, is_simple_sine,
        is_simple_cosine,   is_constant_integral, is_simple_arctan,
    };

    __device__ const Application APPLICATIONS[] = {
        integrate_simple_variable, integrate_power_function, integrate_simple_exponent,
        integrate_simple_sine,     integrate_simple_cosine,  integrate_constant_integral,
        integrate_simple_arctan,
    };

#ifdef __CUDA_ARCH__
    __device__
#endif
        const size_t COUNT =
            Util::ensure_same_v<Util::array_len(CHECKS), Util::array_len(APPLICATIONS)>;

    __device__ Symbol& prepare_solution(const Integral& integral, Symbol& destination) {
        Solution* const solution = destination << Solution::builder();
        Symbol::copy_symbol_sequence(Symbol::from(solution->first_substitution()),
                                     Symbol::from(integral.first_substitution()),
                                     integral.substitutions_size());
        solution->seal_substitutions(integral.substitution_count, integral.substitutions_size());

        return *solution->expression();
    }
}
