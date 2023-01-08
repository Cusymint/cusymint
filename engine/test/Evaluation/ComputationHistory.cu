#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "Evaluation/ComputationHistory.cuh"
#include "Evaluation/TransformationType.cuh"
#include "IntegratorUtils.cuh"
#include "Parser/Parser.cuh"
#include "Symbol/Integral.cuh"
#include "Symbol/Macros.cuh"
#include "Symbol/Solution.cuh"
#include "Symbol/SubexpressionCandidate.cuh"
#include "Symbol/SubexpressionVacancy.cuh"
#include "Symbol/Variable.cuh"

#define FIND_INDEX_BY_UID_TEST(_name, _expressions, _uid, _expected_index)   \
    TEST(ComputationStep, _name) {                                           \
        test_find_index_in_tree_by_uid(_expressions, _uid, _expected_index); \
    }

#define EXPECT_EXPR_EQ(_vec1, _vec2)                                                  \
    EXPECT_TRUE(Sym::Symbol::are_expressions_equal(*(_vec1).data(), *(_vec2).data())) \
        << "Unexpected result:\n"                                                     \
        << (_vec1).data()->to_string() << " <- got,\n"                                \
        << (_vec2).data()->to_string() << " <- expected"

namespace Test {
    using Sym::ComputationHistory;
    using Sym::ComputationStep;
    using Sym::ComputationStepType;
    using Sym::TransformationList;
    using ExprVector = std::vector<std::vector<Sym::Symbol>>;
    using StringVector = std::vector<std::string>;
    namespace {
        void test_find_index_in_tree_by_uid(const ExprVector& expressions, size_t uid,
                                            ssize_t expected_index) {
            ComputationStep step(expressions, {}, ComputationStepType::Simplify);
            EXPECT_EQ(step.find_index_in_tree_by_uid(uid), expected_index);
        }

        void test_copy_solution_path_from(const ExprVector& to, const ExprVector& from,
                                          const std::vector<Sym::Symbol>& expected_result) {
            ComputationStep dst(to, {}, ComputationStepType::Simplify);
            ComputationStep src(from, {}, ComputationStepType::Simplify);

            dst.copy_solution_path_from(src);
            const auto result = dst.get_expression();

            EXPECT_EXPR_EQ(result, expected_result);
        }

        std::string to_string_with_tab(const TransformationList& list) {
            std::string result = "[\n";
            for (const auto& elem : list) {
                result += "\t" + elem->get_description() + ",\n";
            }
            return result + "]";
        }

        const ExprVector expressions = {
            Sym::single_integral_vacancy(),
            nth_expression_candidate_with_uid(1, 0, Sym::e() ^ Sym::single_integral_vacancy()),
            nth_expression_candidate_with_uids(5, 1, 1, Sym::integral(Sym::var()), 3),
            nth_expression_candidate_with_uids(7, 1, 0, Sym::integral(Sym::e())),
        };

        const ExprVector solved_expressions = {
            vacancy_solved_by(1),
            nth_expression_candidate_with_uid(1, 0, Sym::e() ^ vacancy_solved_by(2)),
            nth_expression_candidate_with_uids(
                8, 5, 1, Sym::solution(Sym::num(0.5) * (Sym::var() ^ Sym::num(2))), 3),
            nth_expression_candidate_with_uids(9, 7, 0, Sym::solution(Sym::var() * Sym::e())),
        };
    }

    FIND_INDEX_BY_UID_TEST(FindExistentUid, expressions, 5, 2)
    FIND_INDEX_BY_UID_TEST(FindNonExistentUid, expressions, 2, -1)

    TEST(ComputationStep, GetExpression) {
        ComputationStep step(
            {
                vacancy_solved_by(1),
                nth_expression_candidate_with_uid(1, 0, Sym::e() ^ vacancy_solved_by(3)),
                nth_expression_candidate_with_uid(2, 1, Sym::var()),
                nth_expression_candidate_with_uid(
                    5, 1, Sym::solution(Sym::num(0.5) * (Sym::integral(Sym::var()) ^ Sym::num(2))),
                    3),
                nth_expression_candidate_with_uid(7, 0, Sym::solution(Sym::var() * Sym::e())),
            },
            {}, ComputationStepType::ApplyHeuristic);
        EXPECT_EXPR_EQ(step.get_expression(),
                       Sym::e() ^ (Sym::num(0.5) * (Sym::integral(Sym::var()) ^ Sym::num(2))));
    }

    TEST(ComputationStep, CopySolutionPathFrom) {
        test_copy_solution_path_from(expressions, solved_expressions,
                                     Sym::e() ^ Sym::integral(Sym::var()));
    }

    TEST(ComputationHistory, Complete) {
        ComputationHistory history;
        history.add_step({expressions, {}, ComputationStepType::Simplify});
        history.add_step({solved_expressions, {}, ComputationStepType::ApplySolution});

        history.complete();

        EXPECT_TRUE(history.is_completed());

        const auto& solved_steps = history.get_steps();
        const ExprVector expected_steps = {
            Sym::e() ^ Sym::integral(Sym::var()),
            Sym::e() ^ (Sym::num(0.5) * (Sym::var() ^ Sym::num(2))),
        };
        auto solved_steps_it = solved_steps.cbegin();
        auto expected_steps_it = expected_steps.cbegin();

        ASSERT_EQ(solved_steps.size(), expected_steps.size());

        while (solved_steps_it != solved_steps.cend()) {
            EXPECT_EXPR_EQ((*solved_steps_it).get_expression(), *expected_steps_it);
            ++solved_steps_it;
            ++expected_steps_it;
        }
    }

    TEST(ComputationStep, GetOperations) {
        ComputationStep step(expressions, {}, ComputationStepType::Simplify);
        ComputationStep solved_step(solved_expressions, {}, ComputationStepType::ApplySolution);

        step.copy_solution_path_from(solved_step);
        const auto result = solved_step.get_operations(step);

        TransformationList expected_list;

        expected_list.push_back(std::make_unique<Sym::SolveIntegral>(
            Sym::integral(Sym::var()), Sym::num(0.5) * (Sym::var() ^ Sym::num(2)), 0));

        ASSERT_EQ(result.size(), expected_list.size())
            << "Unexpected result:\n"
            << to_string_with_tab(result) << " <- got,\n"
            << to_string_with_tab(expected_list) << " <- expected";

        for (auto it = result.cbegin(), expected_it = expected_list.cbegin(); it != result.cend();
             ++it, ++expected_it) {
            ASSERT_TRUE(it->get()->equals(*expected_it->get()))
                << "Unexpected result:\n"
                << to_string_with_tab(result) << " <- got,\n"
                << to_string_with_tab(expected_list) << " <- expected";
        }
    }

}