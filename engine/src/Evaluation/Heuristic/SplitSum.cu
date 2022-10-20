#include "SplitSum.cuh"

namespace Sym::Heuristic {
    constexpr size_t EXPRESSION_MAX_SYMBOL_COUNT = 256;

    __device__ CheckResult is_sum(const Integral* const integral) {
        if (integral->integrand()->is(Type::Addition)) {
            return {2, 1};
        }

        return {0, 0};
    }

    __device__ void split_sum(const SubexpressionCandidate* const integral,
                              Symbol* const integral_dst, Symbol* const expression_dst,
                              const size_t expression_index, Symbol* const /*help_space*/) {
        const auto& integrand = integral->arg().as<Integral>().integrand()->as<Addition>();

        SubexpressionCandidate* candidate_expression = expression_dst
                                                       << SubexpressionCandidate::builder();
        candidate_expression->copy_metadata_from(*integral);
        candidate_expression->subexpressions_left = 2;
        Addition* addition = &candidate_expression->arg() << Addition::builder();
        addition->arg1().init_from(SubexpressionVacancy::for_single_integral());
        addition->seal_arg1();
        addition->arg2().init_from(SubexpressionVacancy::for_single_integral());
        addition->seal();
        candidate_expression->seal();

        SubexpressionCandidate* first_integral_candidate = integral_dst
                                                           << SubexpressionCandidate::builder();
        first_integral_candidate->vacancy_expression_idx = expression_index;
        first_integral_candidate->vacancy_idx = 2;
        first_integral_candidate->subexpressions_left = 0;
        integral->arg().as<Integral>().copy_without_integrand_to(&first_integral_candidate->arg());
        auto& first_integral = first_integral_candidate->arg().as<Integral>();
        integrand.arg1().copy_to(first_integral.integrand());
        first_integral.seal();
        first_integral_candidate->seal();

        SubexpressionCandidate* second_integral_candidate =
            integral_dst + EXPRESSION_MAX_SYMBOL_COUNT << SubexpressionCandidate::builder();
        second_integral_candidate->vacancy_expression_idx = expression_index;
        second_integral_candidate->vacancy_idx = 3;
        second_integral_candidate->subexpressions_left = 0;
        integral->arg().as<Integral>().copy_without_integrand_to(&second_integral_candidate->arg());
        auto& second_integral = second_integral_candidate->arg().as<Integral>();
        integrand.arg2().copy_to(second_integral.integrand());
        second_integral.seal();
        second_integral_candidate->seal();
    }
}
