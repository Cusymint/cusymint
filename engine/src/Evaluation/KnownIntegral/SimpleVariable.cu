#include "SimpleVariable.cuh"

#include "Symbol/Symbol.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_variable(const Integral* const integral) {
        return integral->integrand()->is(Type::Variable) ? 1 : 0;
    }

    __device__ void integrate_simple_variable(const Integral* const integral,
                                              Symbol* const destination,
                                              Symbol* const /*help_space*/) {
        Symbol* const solution_expr = prepare_solution(integral, destination);

        Product* const product = solution_expr << Product::builder();
        product->arg1().numeric_constant = NumericConstant::with_value(0.5);
        product->seal_arg1();

        Power* const power = &product->arg2() << Power::builder();
        power->arg1().variable = Variable::create();
        power->seal_arg1();
        power->arg2().numeric_constant = NumericConstant::with_value(2.0);
        power->seal();
        product->seal();

        destination->solution.seal();
    }
}
