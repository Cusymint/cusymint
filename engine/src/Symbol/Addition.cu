#include "Addition.cuh"

#include "MetaOperators.cuh"
#include "Symbol.cuh"
#include "Symbol/Macros.cuh"
#include "Symbol/Product.cuh"
#include "Symbol/SimplificationResult.cuh"
#include "TreeIterator.cuh"
#include "Utils/Cuda.cuh"
#include "Utils/Order.cuh"

#include <fmt/core.h>

namespace Sym {
    DEFINE_TWO_ARGUMENT_COMMUTATIVE_OP_FUNCTIONS(Addition)
    DEFINE_SIMPLE_ONE_ARGUMENT_OP_ARE_EQUAL(Addition)
    DEFINE_IDENTICAL_COMPARE_TO(Addition)
    DEFINE_TWO_ARGUMENT_OP_COMPRESS_REVERSE_TO(Addition)

    DEFINE_SIMPLIFY_IN_PLACE(Addition) {
        simplify_structure(help_space);

        if (!symbol().is(Type::Addition)) {
            return true;
        }

        const auto result = simplify_pairs(help_space);
        eliminate_zeros();
        simplify_structure(help_space);
        return !is_another_loop_required(result);
    }

    DEFINE_IS_FUNCTION_OF(Addition) {
        for (size_t i = 0; i < expression_count; ++i) {
            if (!expressions[i]->is(Type::Addition)) {
                continue;
            }

            const auto& addition_expression = expressions[i]->as<Addition>();

            // TODO: In the future, this should look for correspondences in the addition tree of
            // arg1() and arg2() (See the comment in the same function for Power symbol).
            // Although in this case, this might not be important, as checking whether something is
            // a function of `f(x)+g(x)` is quite rare
            if (Symbol::are_expressions_equal(arg1(), addition_expression.arg1()) &&
                Symbol::are_expressions_equal(arg2(), addition_expression.arg2())) {
                return true;
            }
        }

        return arg1().is_function_of(expressions, expression_count) &&
               arg2().is_function_of(expressions, expression_count);
    }

    DEFINE_INSERT_REVERSED_DERIVATIVE_AT(Addition) {
        if ((&destination - 1)->is(0) && (&destination - 2)->is(0)) {
            return -1;
        }
        return Add<None, None>::init_reverse(destination);
    }

    DEFINE_DERIVATIVE_SIZE(Addition) {
        if ((&destination - 1)->is(0) && (&destination - 2)->is(0)) {
            return -1;
        }
        return Add<None, None>::Size::get_value();
    }

    __host__ __device__ bool Addition::is_sine_cosine_squared_sum(const Symbol* const expr1,
                                                                  const Symbol* const expr2) {
        return PatternPair<Pow<Cos<Same>, Integer<2>>, Pow<Sin<Same>, Integer<2>>>::match_pair(
            *expr1, *expr2);
    }

    DEFINE_TRY_FUSE_SYMBOLS(Addition) { // NOLINT(misc-unused-parameters)
        // Check if one of the arguments is not equal to zero so that we don't go into an infinite
        // loop. Zeros are later removed in `eliminate_zeros`
        if (expr1->is(Type::NumericConstant) && expr2->is(Type::NumericConstant) &&
            expr1->as<NumericConstant>().value != 0.0 &&
            expr2->as<NumericConstant>().value != 0.0) {
            expr1->as<NumericConstant>().value += expr2->as<NumericConstant>().value;
            expr2->as<NumericConstant>().value = 0.0;
            return SimplificationResult::Success;
        }

        // TODO: Some other trigonometric identities
        if (is_sine_cosine_squared_sum(expr1, expr2)) {
            expr1->init_from(NumericConstant::with_value(1.0));
            expr2->init_from(NumericConstant::with_value(0.0));
            return SimplificationResult::Success;
        }

        return SimplificationResult::NoAction;
    }

    DEFINE_COMPARE_AND_TRY_FUSE_SYMBOLS(Addition) {
        if (expr1->is(Type::NumericConstant) && expr2->is(Type::NumericConstant)) {
            destination->init_from(NumericConstant::with_value(expr1->as<NumericConstant>().value +
                                                               expr2->as<NumericConstant>().value));
            return Util::Order::Equal;
        }

        double coef1 = 0.0;
        double coef2 = 0.0;
        const Symbol& base1 = extract_base_and_coefficient(*expr1, coef1);
        const Symbol& base2 = extract_base_and_coefficient(*expr2, coef2);
        const auto order = Symbol::compare_expressions(base1, base2, *destination);

        if constexpr (COMPARE_ONLY) {
            return order;
        }

        if (order != Util::Order::Equal) {
            return order;
        }

        const double sum = coef1 + coef2;
        if (sum == 0) {
            destination->init_from(NumericConstant::with_value(0));
        }
        else if (sum == 1) {
            base1.copy_to(*destination);
        }
        else {
            Mul<Num, Copy>::init(*destination, {coef1 + coef2, base1});
        }
        return Util::Order::Equal;
    }

    __host__ __device__ const Sym::Symbol&
    Addition::extract_base_and_coefficient(const Sym::Symbol& symbol, double& coefficient) {
        if (Sym::Mul<Sym::Num, Sym::Any>::match(symbol)) {
            coefficient = symbol.as<Sym::Product>().arg1().as<Sym::NumericConstant>().value;
            const Sym::Symbol& base = symbol.as<Sym::Product>().arg2();
            return base;
        }

        coefficient = 1;
        return symbol;
    }

    __host__ __device__ double Addition::coefficient(const Sym::Symbol& symbol) {
        for (ConstTreeIterator<Product> iterator(&symbol); iterator.is_valid();
             iterator.advance()) {
            if (iterator.current()->is(Type::NumericConstant)) {
                return iterator.current()->as<NumericConstant>().value;
            }
        }

        return 1;
    }

    __host__ __device__ bool Addition::are_equal_except_for_constant(const Sym::Symbol& expr1,
                                                                     const Sym::Symbol& expr2) {
        ConstTreeIterator<Product> it1(&expr1);
        ConstTreeIterator<Product> it2(&expr2);

        while (it1.is_valid() && it2.is_valid()) {
            if (Symbol::are_expressions_equal(*it1.current(), *it2.current())) {
                it1.advance();
                it2.advance();
                continue;
            }

            if (it1.current()->is(Type::NumericConstant) &&
                it2.current()->is(Type::NumericConstant)) {
                it1.advance();
                it2.advance();
                continue;
            }

            if (it1.current()->is(Type::NumericConstant)) {
                it1.advance();
                continue;
            }

            if (it2.current()->is(Type::NumericConstant)) {
                it2.advance();
                continue;
            }

            return false;
        }

        return !it1.is_valid() && !it2.is_valid();
    }

    __host__ __device__ bool Addition::are_equal_of_opposite_sign(const Symbol& expr1,
                                                                  const Symbol& expr2) {
        return are_equal_except_for_constant(expr1, expr2) &&
               coefficient(expr1) == -coefficient(expr2);
    }

    __host__ __device__ void Addition::eliminate_zeros() {
        for (auto* last = last_in_tree(); last >= this;
             last = (&last->symbol() - 1)->as_ptr<Addition>()) {
            if (last->arg2().is(Type::NumericConstant) &&
                last->arg2().as<NumericConstant>().value == 0.0) {
                last->arg1().move_to(last->symbol());
                continue;
            }

            if (last->arg1().is(Type::NumericConstant) &&
                last->arg1().as<NumericConstant>().value == 0.0) {
                last->arg2().move_to(last->symbol());
            }
        }
    }

    std::string Addition::to_string() const {
        return fmt::format("({}+{})", arg1().to_string(), arg2().to_string());
    }

    std::string Addition::to_tex() const {
        if (arg2().is(Type::NumericConstant) && arg2().as<NumericConstant>().value < 0.0) {
            return fmt::format("{}-{}", arg1().to_tex(), -arg2().as<NumericConstant>().value);
        }

        return fmt::format("{}+{}", arg1().to_tex(), arg2().to_tex());
    }

    std::vector<Symbol> operator+(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs) {
        std::vector<Symbol> res(lhs.size() + rhs.size() + 1);
        Addition::create(lhs.data(), rhs.data(), res.data());
        return res;
    }

    std::vector<Symbol> operator-(const std::vector<Symbol>& arg) {
        std::vector<Symbol> res(Neg<Copy>::size_with({*arg.data()}));
        Neg<Copy>::init(*res.data(), {*arg.data()});
        return res;
    }

    std::vector<Symbol> operator-(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs) {
        return lhs + (-rhs);
    }
}
