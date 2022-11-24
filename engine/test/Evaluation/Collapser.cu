#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "Evaluation/Collapser.cuh"
#include "IntegratorUtils.cuh"
#include "Parser/Parser.cuh"
#include "Symbol/SubexpressionVacancy.cuh"
#include "Symbol/Symbol.cuh"

#define COLLAPSER_TEST(_name) TEST(Collapser, _name)

#define REPLACE_NTH_WITH_TREE_TEST(_name, _expression, _n, _tree, _expected_expression) \
    COLLAPSER_TEST(_name) {                                                             \
        test_replace_nth_with_tree(_expression, _n, _tree, _expected_expression);       \
    }

#define COLLAPSE_NTH_TEST(_name, _expr_vector, _n, _expected_expression) \
    COLLAPSER_TEST(_name) { test_collapse_nth(_expr_vector, _n, _expected_expression); }

#define COLLAPSE_TEST(_name, _expr_vector, _expected_expression) \
    COLLAPSER_TEST(_name) { test_collapse(_expr_vector, _expected_expression); }

namespace Test {
    namespace {
        void test_replace_nth_with_tree(SymVector expression, const size_t n, const SymVector& tree,
                                        const SymVector& expected_result) {
            auto result = Sym::Collapser::replace_nth_with_tree(expression, n, tree);
            EXPECT_TRUE(Sym::Symbol::are_expressions_equal(*result.data(), *expected_result.data()))
                << "Unexpected result:\n " << result.data()->to_string() << " <- got,\n "
                << expected_result.data()->to_string() << " <- expected";
        }

        void test_replace_nth_with_tree(std::string expression, const size_t n, std::string tree,
                                        std::string expected_result) {
            test_replace_nth_with_tree(
                Parser::parse_function(expression), n,
                Sym::first_expression_candidate(Parser::parse_function(tree)),
                Parser::parse_function(expected_result));
        }

        void test_replace_nth_with_tree(SymVector expression, const size_t n, const SymVector& tree,
                                        std::string expected_result) {
            test_replace_nth_with_tree(expression, n, tree,
                                       Parser::parse_function(expected_result));
        }

        void test_collapse_nth(const ExprVector& tree, const size_t n,
                               const SymVector& expected_expression) {
            auto result = Sym::Collapser::collapse_nth(tree, n);
            EXPECT_TRUE(
                Sym::Symbol::are_expressions_equal(*result.data(), *expected_expression.data()))
                << "Unexpected result for collapsing tree with (" << n << ") as root: " << to_string_with_tab(tree) << ":\n "
                << result.data()->to_string() << " <- got,\n "
                << expected_expression.data()->to_string() << " <- expected";
        }

        void test_collapse_nth(const ExprVector& tree, const size_t n,
                               const std::string& expected_expression) {
            test_collapse_nth(tree, n, Parser::parse_function(expected_expression));
        }

        void test_collapse(const ExprVector& tree, const SymVector& expected_expression) {
            auto result = Sym::Collapser::collapse(tree);
            EXPECT_TRUE(
                Sym::Symbol::are_expressions_equal(*result.data(), *expected_expression.data()))
                << "Unexpected result for collapsing tree" << to_string_with_tab(tree) << ":\n "
                << result.data()->to_string() << " <- got,\n "
                << expected_expression.data()->to_string() << " <- expected";
        }

        void test_collapse(const ExprVector& tree, const std::string& expected_expression) {
            test_collapse(tree, Parser::parse_function(expected_expression));
        }

        static ExprVector tree_to_collapse = {
            vacancy_solved_by(3),
            vacancy_solved_by(2),
            nth_expression_candidate(1, "e^x+1"),
            nth_expression_candidate(0, vacancy_solved_by(4) + vacancy_solved_by(5)),
            nth_expression_candidate(3, Sym::sin(vacancy_solved_by(7)), 2),
            nth_expression_candidate(3, "x^2", 3),
            Sym::single_integral_vacancy(),
            nth_expression_candidate(4, vacancy_solved_by(8) * (vacancy_solved_by(9) ^ Sym::pi())),
            nth_expression_candidate(7, "2*(e+ln(x))", 2),
            nth_expression_candidate(7, "arctan(x*(1/x))", 4),
        };
    }

    REPLACE_NTH_WITH_TREE_TEST(ReplaceShortExpression, "x*3*c", 2, "sin(x)^2+5", "(sin(x)^2+5)*3*c")
    REPLACE_NTH_WITH_TREE_TEST(ReplaceLongExpression, "ln(x)+e^e^sin(x)+arcsin(e/3)+1", 5, "0",
                               "ln(x)+0+arcsin(e/3)+1")
    REPLACE_NTH_WITH_TREE_TEST(ReplaceTermGroup, "1+2+3+4+5", 2, "7+8", "7+8+4+5")
    REPLACE_NTH_WITH_TREE_TEST(ReplaceFirstTerm, "x+y", 1, "z+1", "z+1+y")
    REPLACE_NTH_WITH_TREE_TEST(ReplaceLastTerm, "x+y", 2, "z+1", "x+(z+1)")

    REPLACE_NTH_WITH_TREE_TEST(
        ExpandSolutionAndReplace, Sym::sin(Sym::var()) ^ Sym::var(), 3,
        Sym::first_expression_candidate(Sym::solution(Sym::sin(Sym::var()),
                                                      {
                                                          Sym::var() + Sym::pi(),
                                                          Sym::ln(Sym::var() * Sym::num(2)),
                                                          (Sym::e() ^ Sym::var()) + Sym::var(),
                                                      })),
        "sin(x)^(sin(e^ln((x+pi)*2)+ln((x+pi)*2)))")

    COLLAPSE_NTH_TEST(CollapseSmallTree, tree_to_collapse, 1, "e^x+1")
    COLLAPSE_NTH_TEST(CollapseChild, tree_to_collapse, 4,
                      nth_expression_candidate(
                          3, Parser::parse_function("sin((2*(e+ln(x)))*arctan(x*(1/x))^pi)"), 2))
    COLLAPSE_NTH_TEST(CollapseWholeTree, tree_to_collapse, 0,
                      "sin((2*(e+ln(x)))*arctan(x*(1/x))^pi)+x^2")

    COLLAPSE_TEST(CollapseAndSimplifyTree, tree_to_collapse, "x^2+sin((2*(e+ln(x)))*arctan(1)^pi)")
}