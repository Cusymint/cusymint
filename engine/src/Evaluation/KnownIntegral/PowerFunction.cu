#include "PowerFunction.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_power_function(const Integral* const integral) {
        const Symbol* const integrand = integral->integrand();
        if (!integrand[0].is(Type::Power) || !integrand[1].is(Type::Variable)) {
            return 0;
        }

        if (integrand[2].is(Type::NumericConstant) &&
            integrand[2].as<NumericConstant>().value == -1.0) {
            return 0;
        }

        return integrand[2].is_constant() ? 1 : 0;
    }

    __device__ void integrate_power_function(const Integral* const integral,
                                             Symbol* const destination,
                                             Symbol* const /*help_space*/) {
        const Symbol* const integrand = integral->integrand();

        Symbol* const solution_expr = prepare_solution(integral, destination);
        const Symbol* const exponent = &integral->integrand()->power.arg2();

        // `1/(c+1) * x^(c+1)`, `c` can be a whole expression
        Product* const product = solution_expr << Product::builder();

        Reciprocal* const reciprocal = &product->arg1() << Reciprocal::builder();
        Addition* const multiplier_addition = &reciprocal->arg() << Addition::builder();
        exponent->copy_to(&multiplier_addition->arg1());
        multiplier_addition->seal_arg1();
        multiplier_addition->arg2().numeric_constant = NumericConstant::with_value(1.0);
        multiplier_addition->seal();
        reciprocal->seal();
        product->seal_arg1();

        Power* const power = &product->arg2() << Power::builder();
        power->arg1().variable = Variable::create();
        power->seal_arg1();
        Addition* const exponent_addition = &power->arg2() << Addition::builder();
        exponent->copy_to(&exponent_addition->arg1());
        exponent_addition->seal_arg1();
        exponent_addition->arg2().numeric_constant = NumericConstant::with_value(1.0);
        exponent_addition->seal();
        power->seal();
        product->seal();

        destination->solution.seal();
    }
}
