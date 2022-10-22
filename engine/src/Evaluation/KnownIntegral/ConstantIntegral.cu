#include "ConstantIntegral.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_constant_integral(const Integral& integral) {
        const Symbol* const integrand = integral.integrand();
        return integrand->is_constant() ? 1 : 0;
    }

    __device__ void integrate_constant_integral(const Integral& integral, Symbol& destination,
                                                Symbol& /*help_space*/) {
        const Symbol* const integrand = integral.integrand();
        Symbol& solution_expr = prepare_solution(integral, destination);

        Product* const product = solution_expr << Product::builder();
        product->arg1().variable = Variable::create();
        product->seal_arg1();
        integrand->copy_to(&product->arg2());
        product->seal();

        destination.solution.seal();
    }
}
