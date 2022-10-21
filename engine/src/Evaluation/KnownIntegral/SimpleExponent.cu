#include "SimpleExponent.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_exponent(const Integral* const integral) {
        const Symbol* const integrand = integral->integrand();
        return integrand[0].is(Type::Power) && integrand[1].is(Type::KnownConstant) &&
                       integrand[1].as<KnownConstant>().value == KnownConstantValue::E &&
                       integrand[2].is(Type::Variable)
                   ? 1
                   : 0;
    }

    __device__ void integrate_simple_exponent(const Integral* const integral,
                                              Symbol* const destination,
                                              Symbol* const /*help_space*/) {
        Symbol* const solution_expr = prepare_solution(integral, destination);
        const Symbol* const integrand = integral->integrand();

        Power* const power = solution_expr << Power::builder();
        power->arg1().known_constant = KnownConstant::with_value(KnownConstantValue::E);
        power->seal_arg1();
        power->arg2().variable = Variable::create();
        power->seal();

        destination->solution.seal();
    }
}
