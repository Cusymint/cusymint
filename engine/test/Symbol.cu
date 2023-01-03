#include <gtest/gtest.h>

#include "Evaluation/Integrator.cuh"
#include "Symbol/MetaOperators.cuh"
#include "Symbol/Symbol.cuh"

#include "Parser/Parser.cuh"

namespace Test {
    class SymbolCompare : public ::testing::Test {
        static const size_t HELP_SPACE_SIZE = 512;

      protected:
        std::vector<Sym::Symbol> help_space;
        SymbolCompare() : help_space(HELP_SPACE_SIZE) {}
    };

    TEST_F(SymbolCompare, EqualExpressionsEqual) {
        const auto expr1 = Parser::parse_function("x+cos(e^x-a+10)-123+tan(x*x)");
        const auto expr2 = expr1;

        EXPECT_EQ(Util::Order::Equal, Sym::Symbol::compare_expressions(*expr1.data(), *expr2.data(),
                                                                       *help_space.data()));
    }

    TEST_F(SymbolCompare, DifferentExpressionsNotEqual) {
        const auto expr1 = Parser::parse_function("x+cos(e^x-a+10)-123+tan(x*x)");
        const auto expr2 = Parser::parse_function("cos(e^x)+10-20");

        EXPECT_NE(Util::Order::Equal, Sym::Symbol::compare_expressions(*expr1.data(), *expr2.data(),
                                                                       *help_space.data()));
    }

    TEST_F(SymbolCompare, ReverseOrder) {
        const auto expr1 = Parser::parse_function("tan(10)+x");
        const auto expr2 = Parser::parse_function("sin(x^x)");

        const auto order =
            Sym::Symbol::compare_expressions(*expr1.data(), *expr2.data(), *help_space.data());
        EXPECT_NE(Util::Order::Equal, order);

        const auto reverse_order =
            order == Util::Order::Less ? Util::Order::Greater : Util::Order::Less;

        EXPECT_EQ(reverse_order, Sym::Symbol::compare_expressions(*expr2.data(), *expr1.data(),
                                                                  *help_space.data()));
    }

    TEST(Symbol, AlmostConstant) {
        ASSERT_EQ(Parser::parse_function("10+pi*e+sgn(e^(-x^2))").data()->is_almost_constant(),
                  true);
    }
}
