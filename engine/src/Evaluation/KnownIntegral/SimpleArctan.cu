#include "SimpleArctan.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_arctan(const Integral& integral) {
        const Symbol* const integrand = integral.integrand();
        // 1/(x^2+1) or 1/(1+x^2)
        return integrand[0].is(Type::Product) && integrand[1].is(Type::NumericConstant) &&
                       integrand[1].numeric_constant.value == 1.0 &&
                       integrand[2].is(Type::Reciprocal) && integrand[3].is(Type::Addition) &&
                       ((integrand[4].is(Type::Power) && integrand[5].is(Type::Variable) &&
                         integrand[6].is(Type::NumericConstant) &&
                         integrand[6].numeric_constant.value == 2.0 &&
                         integrand[7].is(Type::NumericConstant) &&
                         integrand[7].numeric_constant.value == 1.0) ||
                        (integrand[4].is(Type::NumericConstant) &&
                         integrand[4].numeric_constant.value == 1.0 &&
                         integrand[5].is(Type::Power) && integrand[6].is(Type::Variable) &&
                         integrand[7].is(Type::NumericConstant) &&
                         integrand[7].numeric_constant.value == 2.0))
                   ? 1
                   : 0;
    }

    __device__ void integrate_simple_arctan(const Integral& integral, Symbol& destination,
                                            Symbol& /*help_space*/) {
        const Symbol* const integrand = integral.integrand();
        Symbol& solution_expr = prepare_solution(integral, destination);

        Arctangent* const arctangent = solution_expr << Arctangent::builder();
        arctangent->arg().variable = Variable::create();
        arctangent->seal();

        destination.solution.seal();
    }
}
