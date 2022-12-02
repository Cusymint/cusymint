#include <gtest/gtest.h>

#include <string>
#include <tuple>
#include <vector>

#include "Evaluation/Integrator.cuh"
#include "Parser/Parser.cuh"
#include "Symbol/Integral.cuh"
#include "Symbol/Macros.cuh"
#include "Symbol/MetaOperators.cuh"
#include "Symbol/Product.cuh"
#include "Symbol/Solution.cuh"
#include "Symbol/SubexpressionCandidate.cuh"
#include "Symbol/SubexpressionVacancy.cuh"
#include "Symbol/Symbol.cuh"
#include "Symbol/Variable.cuh"

#define _META_TEST_MATCH(_name, _pattern, _expression, _should_match)      \
    TEST(MetaOperatorsMatchTest, _name) {                                  \
        test_meta_match<MACRO_TYPE(_pattern), _should_match>(_expression); \
    }

#define META_TEST_MATCH(_name, _pattern, _expression) \
    _META_TEST_MATCH(_name, _pattern, _expression, true)

#define META_TEST_NOT_MATCH(_name, _pattern, _expression) \
    _META_TEST_MATCH(_name, _pattern, _expression, false)

#define _META_TEST_MATCH_PAIR(_name, _pattern1, _pattern2, _expression1, _expression2,     \
                              _should_match)                                               \
    TEST(MetaOperatorsMatchTest, _name) {                                                  \
        test_meta_match_pair<MACRO_TYPE(_pattern1), MACRO_TYPE(_pattern2), _should_match>( \
            _expression1, _expression2);                                                   \
    }

#define META_TEST_MATCH_PAIR(_name, _pattern1, _pattern2, _expression1, _expression2) \
    _META_TEST_MATCH_PAIR(_name, _pattern1, _pattern2, _expression1, _expression2, true)

#define META_TEST_NOT_MATCH_PAIR(_name, _pattern1, _pattern2, _expression1, _expression2) \
    _META_TEST_MATCH_PAIR(_name, _pattern1, _pattern2, _expression1, _expression2, false)

#define META_TEST_INIT(_name, _pattern, ...) \
    TEST(MetaOperatorsInitTest, _name) { test_meta_init<MACRO_TYPE(_pattern)>(__VA_ARGS__); }

#define META_TEST_SIZE(_name, _pattern, _size)                       \
    TEST(MetaOperatorsSizeTest, _name) {                             \
        EXPECT_EQ(MACRO_TYPE(_pattern)::Size::get_value(), (_size)); \
    }

#define META_TEST_NO_SIZE(_name, _pattern) \
    TEST(MetaOperatorsSizeTest, _name) { EXPECT_EQ(MACRO_TYPE(_pattern)::Size::HAS_VALUE, false); }

namespace Test {
    using namespace Sym;

    namespace {
        template <class T, bool SHOULD_MATCH>
        void test_meta_match(const std::vector<Symbol>& expression) {
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
        void test_meta_match_pair(const std::vector<Symbol>& expression1,
                                  const std::vector<Symbol>& expression2) {
            if constexpr (SHOULD_MATCH) {
                EXPECT_TRUE(MACRO_TYPE((PatternPair<T1, T2>))::match_pair(*expression1.data(),
                                                                          *expression2.data()));
            }
            else {
                EXPECT_FALSE(MACRO_TYPE((PatternPair<T1, T2>))::match_pair(*expression1.data(),
                                                                           *expression2.data()));
            }
        }

        template <class T1, class T2, bool SHOULD_MATCH>
        void test_meta_match_pair(const std::string& expression1, const std::string& expression2) {
            test_meta_match_pair<T1, T2, SHOULD_MATCH>(Parser::parse_function(expression1),
                                                       Parser::parse_function(expression2));
        }

        template <class T, class... Args>
        void test_meta_init(const std::vector<Symbol>& expected_expression, const Args&... args) {
            std::vector<Symbol> expression(EXPRESSION_MAX_SYMBOL_COUNT);
            T::init(*expression.data(), {args...});
            expression.resize(expression[0].size());
            EXPECT_TRUE(
                Symbol::are_expressions_equal(*expression.data(), *expected_expression.data()))
                << "Expressions do not match:\n"
                << expression.data()->to_string() << " <- got,\n"
                << expected_expression.data()->to_string() << " <- expected\n";
        }

        template <class T, class... Args>
        void test_meta_init(const std::string& expected_expression, const Args&... args) {
            test_meta_init<T, Args...>(Parser::parse_function(expected_expression), args...);
        }
    }

    // Init
    META_TEST_INIT(Variable, Var, "x")
    META_TEST_INIT(Pi, Pi, "pi")
    META_TEST_INIT(E, E, "e")
    META_TEST_INIT(Integer, Integer<69>, "69")
    META_TEST_INIT(NumericConstant, Num, "123.456", 123.456)
    META_TEST_INIT(Copy, Copy, "x^2", *(var() ^ num(2)).data())
    // Simple OneArgOperators
    META_TEST_INIT(Sine, Sin<E>, "sin(e)")
    META_TEST_INIT(Cosine, Cos<Var>, "cos(x)")
    META_TEST_INIT(Tangent, Tan<Pi>, "tan(pi)")
    META_TEST_INIT(Cotangent, Cot<Sin<Var>>, "cot(sin(x))")
    META_TEST_INIT(Arcsine, Arcsin<E>, "arcsin(e)")
    META_TEST_INIT(Arccosine, Arccos<E>, "arccos(e)")
    META_TEST_INIT(Arctangent, Arctan<E>, "arctan(e)")
    META_TEST_INIT(Arccotangent, Arccot<E>, "arccot(e)")
    META_TEST_INIT(Logarithm, Ln<Var>, "ln(x)")
    // Simple TwoArgOperators
    META_TEST_INIT(Sum, (Add<Cos<E>, Pi>), "cos(e)+pi")
    META_TEST_INIT(Product, (Mul<Cos<Var>, Pi>), "cos(x)*pi")
    META_TEST_INIT(Power, (Pow<Cos<E>, Pi>), "cos(e)^pi")
    // Advanced expressions
    META_TEST_INIT(LongSum, (Sum<Var, Cos<Ln<Mul<Num, Var>>>, E, Integer<1>>),
                   "x+(cos(ln(2*x))+(e+1))", 2)
    META_TEST_INIT(LongProduct, (Prod<Add<Var, Num>, Var, Pow<E, Var>>), "(x+5.6)*(x*e^x)", 5.6)
    META_TEST_INIT(EToXTower, (Pow<E, Pow<E, Pow<E, Pow<E, Pow<E, Pow<E, Var>>>>>>),
                   "e^e^e^e^e^e^x")
    // Solution, Candidate, Integral, Vacancy, SingleIntegralVacancy
    META_TEST_INIT(Solution, SolutionOfIntegral<Neg<Cos<Var>>>, solution(-cos(var())),
                   integral(var()).data()->as<Integral>())
    META_TEST_INIT(SubexpressionCandidate, Candidate<Neg<Cos<Var>>>,
                   first_expression_candidate(-cos(var())), (cuda::std::make_tuple(0UL, 0UL, 1)))
    META_TEST_INIT(Integral, Int<Neg<Cos<Var>>>, integral(-cos(var())),
                   integral(var()).data()->as<Integral>())
    META_TEST_INIT(SubexpressionVacancy, Vacancy, single_integral_vacancy(), 0, 1, 0)
    META_TEST_INIT(SingleVacancy, SingleIntegralVacancy, single_integral_vacancy())

    // From::Create::WithMap
    TEST(MetaOperatorsInitTest, FromCreateWithSimpleMap) {
        auto expression = Parser::parse_function("x+4+x^6+e^(2*x)+9+cos(sin(x))+2*u");
        auto expected_expression =
            Parser::parse_function("arcsin(x)*arcsin(4)*arcsin(x^6)*arcsin(e^(2*x))*arcsin(9)*"
                                   "arcsin(cos(sin(x)))*arcsin(2*u)");

        size_t const count = expression.data()->as<Addition>().tree_size();
        std::vector<Symbol> destination(EXPRESSION_MAX_SYMBOL_COUNT);

        From<Addition>::Create<Product>::WithMap<Arcsin>::init(
            *destination.data(), {{expression.data()->as<Addition>(), count}});

        destination.resize(destination.data()->size());

        EXPECT_TRUE(Symbol::are_expressions_equal(*destination.data(), *expected_expression.data()))
            << "Expressions do not match:\n"
            << destination.data()->to_string() << " <- got,\n"
            << expected_expression.data()->to_string() << " <- expected\n";
    }

    template <class T> using Map = Add<Pow<Num, T>, Num>;
    TEST(MetaOperatorsInitTest, FromCreateWithComplexMap) {

        auto expression = Parser::parse_function("x+4+x^6+e^(2*x)+9+cos(sin(x))+2*u");
        auto expected_expression =
            Parser::parse_function("(2^x+3)*(2^4+3)*(2^x^6+3)*(2^e^(2*x)+3)*(2^9+3)*"
                                   "(2^cos(sin(x))+3)*(2^(2*u)+3)");

        size_t const count = expression.data()->as<Addition>().tree_size();
        std::vector<Symbol> destination(EXPRESSION_MAX_SYMBOL_COUNT);

        From<Addition>::Create<Product>::WithMap<Map>::init(
            *destination.data(), {{expression.data()->as<Addition>(), count}, 2, 3});

        destination.resize(destination.data()->size());

        EXPECT_TRUE(Symbol::are_expressions_equal(*destination.data(), *expected_expression.data()))
            << "Expressions do not match:\n"
            << destination.data()->to_string() << " <- got,\n"
            << expected_expression.data()->to_string() << " <- expected\n";
    }

    template <class T> using Map2 = Pow<T, Copy>;
    TEST(MetaOperatorsInitTest, FromCreateWithComplexMapWithCopy) {

        auto expression = Parser::parse_function("x+sin(x)");
        auto to_be_copied = Parser::parse_function("cos(e^x)");
        auto expected_expression =
            Parser::parse_function("x^cos(e^x)*sin(x)^cos(e^x)");

        size_t const count = expression.data()->as<Addition>().tree_size();
        std::vector<Symbol> destination(EXPRESSION_MAX_SYMBOL_COUNT);

        From<Addition>::Create<Product>::WithMap<Map2>::init(
            *destination.data(), {{expression.data()->as<Addition>(), count}, *to_be_copied.data()});

        destination.resize(destination.data()->size());

        EXPECT_TRUE(Symbol::are_expressions_equal(*destination.data(), *expected_expression.data()))
            << "Expressions do not match:\n"
            << destination.data()->to_string() << " <- got,\n"
            << expected_expression.data()->to_string() << " <- expected\n";
    }

    // Match
    META_TEST_MATCH(Variable, Var, "x")
    META_TEST_MATCH(Pi, Pi, "pi")
    META_TEST_MATCH(E, E, "e")
    META_TEST_MATCH(Integer, Integer<69>, "69")
    META_TEST_MATCH(NumericConstant, Num, "123.456")
    // Const
    META_TEST_MATCH(ConstantExpression, Const, "sin(e)+pi+c^456+cos(tan(1))*(-ln(2))")
    META_TEST_NOT_MATCH(NotConstantExpression, Const, "e^pi+sin(x)")
    // Simple OneArgOperators
    META_TEST_MATCH(Sine, Sin<E>, "sin(e)")
    META_TEST_MATCH(Cosine, Cos<Var>, "cos(x)")
    META_TEST_MATCH(Tangent, Tan<Pi>, "tan(pi)")
    META_TEST_MATCH(Cotangent, Cot<Sin<Var>>, "cot(sin(x))")
    META_TEST_MATCH(Arcsine, Arcsin<E>, "arcsin(e)")
    META_TEST_MATCH(Arccosine, Arccos<E>, "arccos(e)")
    META_TEST_MATCH(Arctangent, Arctan<E>, "arctan(e)")
    META_TEST_MATCH(Arccotangent, Arccot<E>, "arccot(e)")
    META_TEST_MATCH(Logarithm, Ln<Var>, "ln(x)")
    // Simple TwoArgOperators
    META_TEST_MATCH(Sum, (Add<Cos<E>, Pi>), "cos(e)+pi")
    META_TEST_MATCH(Product, (Mul<Cos<Var>, Pi>), "cos(x)*pi")
    META_TEST_MATCH(Power, (Pow<Cos<E>, Pi>), "cos(e)^pi")
    // AnyOf, AllOf, Not
    META_TEST_MATCH(AnyOfFirstCorrect, (AnyOf<Cos<Var>, E, Integer<3>>), "cos(x)")
    META_TEST_MATCH(AnyOfSecondCorrect, (AnyOf<Cos<Var>, E, Integer<3>>), "e")
    META_TEST_MATCH(AnyOfLastCorrect, (AnyOf<Cos<Var>, E, Integer<3>>), "3")
    META_TEST_NOT_MATCH(AnyOfNoneCorrect, (AnyOf<Cos<Var>, E, Integer<3>>), "sin(e)")

    META_TEST_NOT_MATCH(NotMatchAllOf, (AllOf<Cos<Var>, E, Integer<3>>), "e")
    META_TEST_MATCH(SingleAllOf, (AllOf<Cos<Var>>), "cos(x)")

    META_TEST_MATCH(NotMatchesWrongExpression, (Not<AllOf<Cos<Var>, Var>>), "cos(x)")
    META_TEST_NOT_MATCH(NotWithTrueCondition, (Not<Arcsin<E>>), "arcsin(e)")
    // Same, PatternPair
    META_TEST_MATCH(SimpleSame, (Mul<Same, Same>), "(e^x*345+1)*(e^x*345+1)")
    META_TEST_NOT_MATCH(NotSame, (Mul<Same, Same>), "(e^x*345+1)*(e^x*345)")
    META_TEST_MATCH(AdvancedSame, (Add<Ln<Mul<Same, Num>>, Sin<Add<E, Same>>>),
                    "ln((x+sin(x)+4^x)*5.7)+sin(e+(x+sin(x)+4^x))")
    META_TEST_MATCH(FourSameSymbols, (Mul<Add<Same, Same>, Add<Same, Mul<Integer<4>, Same>>>),
                    "(e^c^x+e^c^x)*(e^c^x+4*e^c^x)")

    META_TEST_MATCH(SumOfNotSameTerms, (Add<Same, Not<Same>>), "x+y")
    META_TEST_NOT_MATCH(NotMatchSumOfSameTerms, (Add<Same, Not<Same>>), "x+x")

    META_TEST_MATCH(SameWithAllOf, (Add<Same, AllOf<Sin<E>, Sin<Same>>>), "e+sin(e)")
    META_TEST_MATCH(SameWithAnyOf, (Add<Same, AnyOf<Sin<E>, Sin<Same>>>), "22+sin(22)")
    META_TEST_MATCH(SameWithAnyOf2, (Add<Same, AnyOf<Sin<E>, Sin<Same>>>), "22+sin(e)")

    META_TEST_MATCH_PAIR(PairWithIndependentPatterns, Ln<Var>, Arccot<E>, "ln(x)", "arccot(e)")
    META_TEST_MATCH_PAIR(PairTangentCotangent, Tan<Same>, Cot<Same>, "tg(2*x)", "ctg(2*x)")
    // Advanced expressions
    META_TEST_MATCH(LongSum, (Sum<Var, Cos<Ln<Const>>, E, Integer<1>>), "x+(cos(ln(2+3+c))+(e+1))")
    META_TEST_MATCH(LongProduct, (Prod<Add<Var, Num>, Var, Pow<E, Var>>), "(x+5.6)*(x*e^x)")
    META_TEST_MATCH(EToXTower, (Pow<E, Pow<E, Pow<E, Pow<E, Pow<E, Pow<E, Var>>>>>>),
                    "e^e^e^e^e^e^x")
    // Solution, Candidate, Integral, Vacancy, SingleIntegralVacancy
    META_TEST_MATCH(Solution, SolutionOfIntegral<Neg<Cos<Var>>>, solution(-cos(var())))
    META_TEST_MATCH(SubexpressionCandidate, Candidate<Neg<Cos<Var>>>,
                    first_expression_candidate(-cos(var())))
    META_TEST_MATCH(Integral, Int<Neg<Cos<Var>>>, integral(-cos(var())))
    META_TEST_MATCH(SingleVacancy, SingleIntegralVacancy, single_integral_vacancy())

    // Sizes
    META_TEST_SIZE(Variable, Var, 1)
    META_TEST_SIZE(Integer, Integer<10>, 1)
    META_TEST_SIZE(ConstantE, E, 1)
    META_TEST_SIZE(ConstantPi, Pi, 1)
    META_TEST_SIZE(Number, Num, 1)
    META_TEST_SIZE(Vacancy, SingleIntegralVacancy, 1)

    META_TEST_SIZE(SimpleComposite1, (Add<Var, Integer<1>>), 3)
    META_TEST_SIZE(SimpleComposite2, (Sub<Num, Var>), 4)
    META_TEST_SIZE(SimpleComposite3, (Pow<Cos<Frac<Var, Integer<2>>>, Var>), 7)

    META_TEST_SIZE(
        ComplexComposite1,
        (Add<Pow<Cos<Frac<Var, Integer<2>>>, Sub<Pow<Var, Var>, Cos<Frac<Var, Integer<3>>>>>, Var>),
        18)
    META_TEST_SIZE(
        ComplexComposite2,
        (Add<Cos<Sub<SingleIntegralVacancy,
                     Sub<Sub<Cos<Frac<Pow<Integer<8>, Num>, Frac<Var, Var>>>, Var>, Num>>>,
             Cos<Sin<Var>>>),
        24)

    META_TEST_NO_SIZE(SimpleCopy, (Add<Var, Copy>))
    META_TEST_NO_SIZE(DeepCopy, (Add<Cos<Sin<Cos<Sin<Copy>>>>, Cos<Sin<Var>>>))
    META_TEST_NO_SIZE(MatchOnly, (Add<Cos<Pow<Not<Integer<5>>, AnyOf<Var, Num>>>, Cos<Sin<Var>>>))
    META_TEST_NO_SIZE(WithIntegral, (Add<SolutionOfIntegral<Var>, Var>))
    META_TEST_NO_SIZE(Same, (Add<Same, Inv<Same>>))
}
