#include "SplitSum.cuh"

#include "Symbol/MetaOperators.cuh"

namespace Sym::Heuristic {
    __device__ CheckResult is_sum(const Integral& integral) {
        if (integral.integrand().is(Type::Addition)) {
            return {2, 1};
        }

        return {0, 0};
    }

    __device__ EvaluationStatus split_sum(const SubexpressionCandidate& integral,
                                          const ExpressionArray<>::Iterator& integral_dst,
                                          const ExpressionArray<>::Iterator& expression_dst,
                                          Symbol& /*help_space*/) {
        const auto& integrand = integral.arg().as<Integral>().integrand().as<Addition>();

        using ExpressionType = Candidate<Add<SingleIntegralVacancy, SingleIntegralVacancy>>;

        ENSURE_ENOUGH_SPACE(ExpressionType::Size::get_value(), expression_dst);
        ExpressionType::init(*expression_dst,
                             {{integral.vacancy_expression_idx, integral.vacancy_idx, 2}});

        const auto first_integral_dst = integral_dst;
        const auto second_integral_dst = integral_dst + 1;

        const auto& original_integral = integral.arg().as<Integral>();
        const auto& original_sum1 = original_integral.integrand().as<Addition>().arg1();
        const auto& original_sum2 = original_integral.integrand().as<Addition>().arg2();

        ENSURE_ENOUGH_SPACE(original_integral.size - 1 - original_sum2.size(), first_integral_dst);
        Candidate<Int<Copy>>::init(
            *first_integral_dst,
            {{expression_dst.index(), 2, 0}, original_integral, original_sum1});

        ENSURE_ENOUGH_SPACE(original_integral.size - 1 - original_sum1.size(), second_integral_dst);
        Candidate<Int<Copy>>::init(
            *second_integral_dst,
            {{expression_dst.index(), 3, 0}, original_integral, original_sum2});

        return EvaluationStatus::Done;
    }
}
