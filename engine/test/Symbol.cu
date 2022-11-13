#include <gtest/gtest.h>

#include "Symbol/MetaOperators.cuh"
#include "Symbol/Symbol.cuh"

#include "Parser/Parser.cuh"

namespace Test {
    TEST(Symbol, EqualExpressionsEqual) {
        const auto expr1 = Parser::parse_function("x+cos(e^x-a+10)-123+tan(x*x)");
        const auto expr2 = expr1; // NOLINT(performance-unnecessary-copy-initialization)

        EXPECT_EQ(Util::Order::Equal,
                  Sym::Symbol::compare_expressions(*expr1.data(), *expr2.data()));
    }

    TEST(Symbol, DifferentExpressionsNotEqual) {
        const auto expr1 = Parser::parse_function("x+cos(e^x-a+10)-123+tan(x*x)");
        const auto expr2 = Parser::parse_function("cos(e^x)+10-20");

        EXPECT_NE(Util::Order::Equal,
                  Sym::Symbol::compare_expressions(*expr1.data(), *expr2.data()));
    }

    TEST(Symbol, ReverseOrder) {
        const auto expr1 = Parser::parse_function("tan(10)+x");
        const auto expr2 = Parser::parse_function("sin(x^x)");

        const auto order = Sym::Symbol::compare_expressions(*expr1.data(), *expr2.data());
        EXPECT_NE(Util::Order::Equal, order);

        const auto reverse_order =
            order == Util::Order::Less ? Util::Order::Greater : Util::Order::Less;

        EXPECT_EQ(reverse_order, Sym::Symbol::compare_expressions(*expr2.data(), *expr1.data()));
    }
}
