#include <gtest/gtest.h>

#include <vector>

#include "Evaluation/Collapser.cuh"
#include "Parser/Parser.cuh"
#include "Symbol/Constants.cuh"
#include "Symbol/Solution.cuh"
#include "Symbol/SubexpressionCandidate.cuh"
#include "Symbol/Symbol.cuh"

#define COLLAPSER_TEST(_name) TEST(Collapser, _name)

#define REPLACE_NTH_WITH_TREE_TEST(_name, _expression, _n, _tree, _expected_expression) \
    COLLAPSER_TEST(_name) {                                                             \
        test_replace_nth_with_tree(_expression, _n, _tree, _expected_expression);       \
    }

namespace Test {
    namespace {
        void test_replace_nth_with_tree(std::vector<Sym::Symbol> expression, const size_t n,
                                        const std::vector<Sym::Symbol>& tree,
                                        const std::vector<Sym::Symbol>& expected_result) {
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

        void test_replace_nth_with_tree(std::vector<Sym::Symbol> expression, const size_t n,
                                        const std::vector<Sym::Symbol>& tree,
                                        std::string expected_result) {
            test_replace_nth_with_tree(expression, n, tree,
                                       Parser::parse_function(expected_result));
        }
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
}