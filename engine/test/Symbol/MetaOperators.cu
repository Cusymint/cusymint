#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "Evaluation/Integrate.cuh"
#include "Parser/Parser.cuh"
#include "Symbol/MetaOperators.cuh"
#include "Symbol/Product.cuh"
#include "Symbol/Symbol.cuh"
#include "Symbol/Variable.cuh"

// This is a workaround for use of commas in template types in macros
template <class T> struct macro_type;
template <class T, class U> struct macro_type<T(U)> {
    using type = U;
};
#define MACRO_TYPE(_pattern) macro_type<void(_pattern)>::type

#define _META_TEST_MATCH(_name, _pattern, _expression, _should_match)      \
    TEST(MetaOperatorsMatchTest, _name) { test_meta_match<MACRO_TYPE(_pattern), _should_match>(_expression); } // NOLINT

#define META_TEST_MATCH(_name, _pattern, _expression) \
    _META_TEST_MATCH(_name, _pattern, _expression, true)

#define META_TEST_NOT_MATCH(_name, _pattern, _expression) \
    _META_TEST_MATCH(_name, _pattern, _expression, false)

#define _META_TEST_MATCH_PAIR(_name, _pattern1, _pattern2, _expression1, _expression2, _should_match) \
    TEST(MetaOperatorsMatchTest, _name) { test_meta_match_pair<MACRO_TYPE(_pattern1), MACRO_TYPE(_pattern2), _should_match>(_expression1, _expression2); } // NOLINT

#define META_TEST_MATCH_PAIR(_name, _pattern1, _pattern2, _expression1, _expression2) \
    _META_TEST_MATCH_PAIR(_name, _pattern1, _pattern2, _expression1, _expression2, true)

#define META_TEST_NOT_MATCH_PAIR(_name, _pattern1, _pattern2, _expression1, _expression2) \
    _META_TEST_MATCH_PAIR(_name, _pattern1, _pattern2, _expression1, _expression2, false)

#define META_TEST_INIT(_name, _pattern, ...)               \
    TEST(MetaOperatorsInitTest, _name) { test_meta_init<MACRO_TYPE(_pattern)>(__VA_ARGS__); } // NOLINT

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

        template <class T1, class T2, bool SHOULD_MATCH>
        void test_meta_match_pair(const std::vector<Sym::Symbol>& expression1, const std::vector<Sym::Symbol>& expression2) {
            if constexpr (SHOULD_MATCH) {
                EXPECT_TRUE(MACRO_TYPE((Sym::PatternPair<T1, T2>))::match_pair(*expression1.data(), *expression2.data()));
            }
            else {
                EXPECT_FALSE(MACRO_TYPE((Sym::PatternPair<T1, T2>))::match_pair(*expression1.data(), *expression2.data()));
            }
        }

        template <class T1, class T2, bool SHOULD_MATCH>
        void test_meta_match_pair(const std::string& expression1, const std::string& expression2) {
            test_meta_match_pair<T1, T2, SHOULD_MATCH>(Parser::parse_function(expression1), Parser::parse_function(expression2));
        }

        template <class T, class... Args>
        void test_meta_init(const std::vector<Sym::Symbol>& expected_expression,
                            const Args&... args) {
            std::vector<Sym::Symbol> expression(Sym::EXPRESSION_MAX_SYMBOL_COUNT);
            T::init(*expression.data(), {args...});
            expression.resize(expression[0].size());
            EXPECT_TRUE(Sym::Symbol::compare_trees(expression.data(), expected_expression.data()))
                << "Expressions do not match:\n"
                << expression.data()->to_string() << " <- got,\n"
                << expected_expression.data()->to_string() << " <- expected\n";
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
    META_TEST_INIT(Sum, (Sym::Add<Sym::Cos<Sym::E>, Sym::Pi>), "cos(e)+pi")
    META_TEST_INIT(Product, (Sym::Mul<Sym::Cos<Sym::Var>, Sym::Pi>), "cos(x)*pi")
    META_TEST_INIT(Power, (Sym::Pow<Sym::Cos<Sym::E>, Sym::Pi>), "cos(e)^pi")
    // Advanced expressions
    META_TEST_INIT(LongSum, (Sym::Sum<Sym::Var, Sym::Cos<Sym::Ln<Sym::Mul<Sym::Num, Sym::Var>>>, Sym::E, Sym::Integer<1>>), "x+(cos(ln(2*x))+(e+1))", 2)
    META_TEST_INIT(LongProduct, (Sym::Prod<Sym::Add<Sym::Var, Sym::Num>, Sym::Var, Sym::Pow<Sym::E, Sym::Var>>), "(x+5.6)*(x*e^x)", 5.6)
    META_TEST_INIT(EToXTower, (Sym::Pow<Sym::E, Sym::Pow<Sym::E, Sym::Pow<Sym::E, Sym::Pow<Sym::E, Sym::Pow<Sym::E, Sym::Pow<Sym::E, Sym::Var>>>>>>), "e^e^e^e^e^e^x")
    // solution, candidate, integral, vacancy, singleIntegralVacancy
    
    // From::Create::WithMap
    TEST(MetaOperatorsInitTest, FromCreateWithMap) { // NOLINT
        auto expression = Parser::parse_function("x+4+x^6+e^(2*x)+9+cos(sin(x))+2*u");
        auto expected_expression = Parser::parse_function(
            "arcsin(2*u)*arcsin(cos(sin(x)))*arcsin(9)*arcsin(e^(2*x))*arcsin(x^6)*arcsin(4)*arcsin(x)");

        size_t const count = expression.data()->as<Sym::Addition>().tree_size();
        std::vector<Sym::Symbol> destination(Sym::EXPRESSION_MAX_SYMBOL_COUNT);

        Sym::From<Sym::Addition>::Create<Sym::Product>::WithMap<Sym::Arcsine>::init(
            *destination.data(), {{expression.data()->as<Sym::Addition>(), count}});

        destination.resize(destination.data()->size());

        EXPECT_TRUE(Sym::Symbol::compare_trees(destination.data(), expected_expression.data()));
    }

    // Match
    META_TEST_MATCH(Variable, Sym::Var, "x")
    META_TEST_MATCH(Pi, Sym::Pi, "pi")
    META_TEST_MATCH(E, Sym::E, "e")
    META_TEST_MATCH(Integer, Sym::Integer<69>, "69")
    META_TEST_MATCH(NumericConstant, Sym::Num, "123.456")
    // Const
    META_TEST_MATCH(ConstantExpression, Sym::Const, "sin(e)+pi+c^456+cos(tan(1))*(-ln(2))")
    META_TEST_NOT_MATCH(NotConstantExpression, Sym::Const, "e^pi+sin(x)")
    // Simple OneArgOperators
    META_TEST_MATCH(Sine, Sym::Sin<Sym::E>, "sin(e)")
    META_TEST_MATCH(Cosine, Sym::Cos<Sym::Var>, "cos(x)")
    META_TEST_MATCH(Tangent, Sym::Tan<Sym::Pi>, "tan(pi)")
    META_TEST_MATCH(Cotangent, Sym::Cot<Sym::Sin<Sym::Var>>, "cot(sin(x))")
    META_TEST_MATCH(Arcsine, Sym::Arcsin<Sym::E>, "arcsin(e)")
    META_TEST_MATCH(Arccosine, Sym::Arccos<Sym::E>, "arccos(e)")
    META_TEST_MATCH(Arctangent, Sym::Arctan<Sym::E>, "arctan(e)")
    META_TEST_MATCH(Arccotangent, Sym::Arccot<Sym::E>, "arccot(e)")
    META_TEST_MATCH(Logarithm, Sym::Ln<Sym::Var>, "ln(x)")
    // Simple TwoArgOperators
    META_TEST_MATCH(Sum, (Sym::Add<Sym::Cos<Sym::E>, Sym::Pi>), "cos(e)+pi")
    META_TEST_MATCH(Product, (Sym::Mul<Sym::Cos<Sym::Var>, Sym::Pi>), "cos(x)*pi")
    META_TEST_MATCH(Power, (Sym::Pow<Sym::Cos<Sym::E>, Sym::Pi>), "cos(e)^pi")
    // AnyOf, AllOf, Not
    META_TEST_MATCH(AnyOfFirstCorrect, (Sym::AnyOf<Sym::Cos<Sym::Var>, Sym::E, Sym::Integer<3>>), "cos(x)")
    META_TEST_MATCH(AnyOfSecondCorrect, (Sym::AnyOf<Sym::Cos<Sym::Var>, Sym::E, Sym::Integer<3>>), "e")
    META_TEST_MATCH(AnyOfLastCorrect, (Sym::AnyOf<Sym::Cos<Sym::Var>, Sym::E, Sym::Integer<3>>), "3")
    META_TEST_NOT_MATCH(AnyOfNoneCorrect, (Sym::AnyOf<Sym::Cos<Sym::Var>, Sym::E, Sym::Integer<3>>), "sin(e)")

    META_TEST_NOT_MATCH(NotMatchAllOf, (Sym::AllOf<Sym::Cos<Sym::Var>, Sym::E, Sym::Integer<3>>), "e")
    META_TEST_MATCH(SingleAllOf, (Sym::AllOf<Sym::Cos<Sym::Var>>), "cos(x)")

    META_TEST_MATCH(NotMatchesWrongExpression, (Sym::Not<Sym::AllOf<Sym::Cos<Sym::Var>, Sym::Var>>), "cos(x)")
    META_TEST_NOT_MATCH(NotWithTrueCondition, (Sym::Not<Sym::Arcsin<Sym::E>>), "arcsin(e)")
    // Same, PatternPair
    META_TEST_MATCH(SimpleSame, (Sym::Mul<Sym::Same, Sym::Same>), "(e^x*345+1)*(e^x*345+1)")
    META_TEST_NOT_MATCH(NotSame, (Sym::Mul<Sym::Same, Sym::Same>), "(e^x*345+1)*(e^x*345)")
    META_TEST_MATCH(AdvancedSame, (Sym::Add<Sym::Ln<Sym::Mul<Sym::Same, Sym::Num>>, Sym::Sin<Sym::Add<Sym::E, Sym::Same>>>), "ln((x+sin(x)+4^x)*5.7)+sin(e+(x+sin(x)+4^x))")
    META_TEST_MATCH(FourSameSymbols, (Sym::Mul<Sym::Add<Sym::Same, Sym::Same>, Sym::Add<Sym::Same, Sym::Mul<Sym::Integer<4>, Sym::Same>>>), "(e^c^x+e^c^x)*(e^c^x+4*e^c^x)")

    META_TEST_MATCH_PAIR(PairWithIndependentPatterns, Sym::Ln<Sym::Var>, Sym::Arccot<Sym::E>, "ln(x)", "arccot(e)")
    // Advanced expressions
    META_TEST_MATCH(LongSum, (Sym::Sum<Sym::Var, Sym::Cos<Sym::Ln<Sym::Const>>, Sym::E, Sym::Integer<1>>), "x+(cos(ln(2+3+c))+(e+1))")
    META_TEST_MATCH(LongProduct, (Sym::Prod<Sym::Add<Sym::Var, Sym::Num>, Sym::Var, Sym::Pow<Sym::E, Sym::Var>>), "(x+5.6)*(x*e^x)")
    META_TEST_MATCH(EToXTower, (Sym::Pow<Sym::E, Sym::Pow<Sym::E, Sym::Pow<Sym::E, Sym::Pow<Sym::E, Sym::Pow<Sym::E, Sym::Pow<Sym::E, Sym::Var>>>>>>), "e^e^e^e^e^e^x")
        
    // solution, candidate, integral, vacancy, singleIntegralVacancy
    
}
