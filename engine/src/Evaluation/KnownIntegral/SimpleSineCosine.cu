#include "SimpleSineCosine.cuh"

namespace Sym::KnownIntegral {
    __device__ size_t is_simple_sine(const Integral* const integral) {
        const Symbol* const integrand = integral->integrand();
        return integrand[0].is(Type::Sine) && integrand[1].is(Type::Variable) ? 1 : 0;
    }

    __device__ void integrate_simple_sine(const Integral* const integral, Symbol* const destination,
                                          Symbol* const /*help_space*/) {
        Symbol* const solution_expr = prepare_solution(integral, destination);
        const Symbol* const integrand = integral->integrand();

        Negation* const minus = solution_expr << Negation::builder();
        Cosine* const cos = &minus->arg() << Cosine::builder();
        cos->arg().variable = Variable::create();
        cos->seal();
        minus->seal();

        destination->solution.seal();
    }

    __device__ size_t is_simple_cosine(const Integral* const integral) {
        const Symbol* const integrand = integral->integrand();
        return integrand[0].is(Type::Cosine) && integrand[1].is(Type::Variable) ? 1 : 0;
    }

    __device__ void integrate_simple_cosine(const Integral* const integral,
                                            Symbol* const destination,
                                            Symbol* const /*help_space*/) {
        Symbol* const solution_expr = prepare_solution(integral, destination);
        const Symbol* const integrand = integral->integrand();

        Sine* const sine = solution_expr << Sine::builder();
        sine->arg().variable = Variable::create();
        sine->seal();

        destination->solution.seal();
    }

}
