#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "Evaluation/Integrate.cuh"
#include "Parser/Parser.cuh"
#include "Symbol/MetaOperators.cuh"
#include "Symbol/Symbol.cuh"
#include "Symbol/Variable.cuh"

#define I ,

#define _META_TEST_MATCH(_name, _pattern, _expression, _should_match) \
    TEST(MetaOperatorsMatchTest, _name) { test_meta_match<_pattern, _should_match>(_expression); } // NOLINT

#define META_TEST_MATCH(_name, _pattern, _expression) \
    _META_TEST_MATCH(_name, _pattern, _expression, true)

#define META_TEST_NOT_MATCH(_name, _pattern, _expression) \
    _META_TEST_MATCH(_name, _pattern, _expression, false)

#define META_TEST_INIT(_name, _pattern, ...) \
    TEST(MetaOperatorsInitTest, _name) { test_meta_init<_pattern>(__VA_ARGS__); } // NOLINT

namespace Test {
    namespace {
        template <class T, bool SHOULD_MATCH>
        void test_meta_match(const std::vector<Sym::Symbol>& expression) {
            if constexpr (SHOULD_MATCH) {
                EXPECT_TRUE(T::match(*expression.data()));
            }
            else {
                EXPECT_FALSE(T::match(*expression.data()));
            }
        }

        template <class T, bool SHOULD_MATCH> void test_meta_match(const std::string& expression) {
            test_meta_match<T, SHOULD_MATCH>(Parser::parse_function(expression));
        }

        template <class T, class... Args>
        void test_meta_init(const std::vector<Sym::Symbol>& expected_expression, const Args&... args) {
            std::vector<Sym::Symbol> expression(Sym::EXPRESSION_MAX_SYMBOL_COUNT);
            T::init(*expression.data(), {args...});
            expression.resize(expression[0].size());
            EXPECT_TRUE(Sym::Symbol::compare_trees(expression.data(), expected_expression.data()));
        }

        template <class T, class... Args>
        void test_meta_init(const std::string& expected_expression, const Args&... args) {
            test_meta_init<T, Args...>(Parser::parse_function(expected_expression), args...);
        }
    }

    META_TEST_INIT(Variable, Sym::Var, "x")
    META_TEST_INIT(Pi, Sym::Pi, "pi")
    META_TEST_INIT(E, Sym::E, "e")
    META_TEST_INIT(Integer, Sym::Integer<69>, "69")
    META_TEST_INIT(NumericConstant, Sym::Num, "123.456", 123.456)
    META_TEST_INIT(Copy, Sym::Copy, "x^2", *(Sym::var() ^ Sym::num(2)).data())
    // Simple OneArgOperators
    META_TEST_INIT(Sine, Sym::Sin<Sym::E>, "sin(e)")
    META_TEST_INIT(Cosine, Sym::Cos<Sym::Var>, "cos(x)")
    META_TEST_INIT(Tangent, Sym::Tan<Sym::Pi>, "tan(pi)")
    META_TEST_INIT(Cotangent, Sym::Cot<Sym::Sin<Sym::Var>>, "cot(sin(x))")
    META_TEST_INIT(Arcsine, Sym::Arcsin<Sym::E>, "arcsin(e)")
    META_TEST_INIT(Arccosine, Sym::Arccos<Sym::E>, "arccos(e)")
    META_TEST_INIT(Arctangent, Sym::Arctan<Sym::E>, "arctan(e)")
    META_TEST_INIT(Arccotangent, Sym::Arccot<Sym::E>, "arccot(e)")
    META_TEST_INIT(Logarithm, Sym::Ln<Sym::Var>, "ln(x)")
    // Simple TwoArgOperators
    META_TEST_INIT(Sum, Sym::Add<Sym::Cos<Sym::E> I Sym::Pi>, "cos(e)+pi")
    // Advanced expressions

    // solution, candidate, integral, vacancy, singleIntegralVacancy
    // from create withmap


    META_TEST_MATCH(Variable, Sym::Var, "x")
}