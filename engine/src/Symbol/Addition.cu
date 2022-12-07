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
        if ((destination - 1)->is(0) && (destination - 2)->is(0)) {
            return -1;
        }
        return Add<None, None>::init_reverse(*destination);
    }

    __host__ __device__ bool Addition::is_sine_cosine_squared_sum(const Symbol* const expr1,
                                                                  const Symbol* const expr2) {
        return PatternPair<Pow<Cos<Same>, Integer<2>>, Pow<Sin<Same>, Integer<2>>>::match_pair(
            *expr1, *expr2);
    }

    __host__ __device__ bool Addition::are_equal_of_opposite_sign(const Symbol& expr1,
                                                                  const Symbol& expr2) {
        return PatternPair<Neg<Same>, Same>::match_pair(expr1, expr2) ||
               PatternPair<Same, Neg<Same>>::match_pair(expr1, expr2);
    }

    DEFINE_TRY_FUSE_SYMBOLS(Addition) { // NOLINT(misc-unused-parameters)
        // Sprawdzenie, czy jeden z argumentów nie jest rónwy zero jest wymagane by nie wpaść w
        // nieskończoną pętlę, zera i tak są potem usuwane w `eliminate_zeros`.
        if (expr1->is(Type::NumericConstant) && expr2->is(Type::NumericConstant) &&
            expr1->as<NumericConstant>().value != 0.0 &&
            expr2->as<NumericConstant>().value != 0.0) {
            expr1->as<NumericConstant>().value += expr2->as<NumericConstant>().value;
            expr2->as<NumericConstant>().value = 0.0;
            return SimplificationResult::Success;
        }

        if (are_equal_of_opposite_sign(*expr1, *expr2)) {
            expr1->init_from(NumericConstant::with_value(0.0));
            expr2->init_from(NumericConstant::with_value(0.0));
            return SimplificationResult::Success;
        }

        // TODO: Jakieś inne tożsamości trygonometryczne
        if (is_sine_cosine_squared_sum(expr1, expr2)) {
            expr1->init_from(NumericConstant::with_value(1.0));
            expr2->init_from(NumericConstant::with_value(0.0));
            return SimplificationResult::Success;
        }

        // TODO: Jedynka hiperboliczna

        return SimplificationResult::NoAction;
    }

    DEFINE_COMPARE_AND_TRY_FUSE_SYMBOLS(Addition) {
        double coef1 = 0.0;
        double coef2 = 0.0;
        const Symbol& base1 = extract_base_and_coefficient(*expr1, coef1);
        const Symbol& base2 = extract_base_and_coefficient(*expr2, coef2);

        const auto order = Symbol::compare_expressions(base1, base2, *destination);

        if (base1.is(Type::NumericConstant) && base2.is(Type::NumericConstant)) {
            destination->init_from(
                NumericConstant::with_value(coef1 * base1.as<NumericConstant>().value +
                                            coef2 * base2.as<NumericConstant>().value));
            return Util::Order::Equal;
        }

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
        else if (sum == -1) {
            Neg<Copy>::init(*destination, {base1});
        }
        else {
            Mul<Num, Copy>::init(*destination, {coef1 + coef2, base1});
        }
        return Util::Order::Equal;
    }

    __host__ __device__ const Sym::Symbol&
    Addition::extract_base_and_coefficient(const Sym::Symbol& symbol, double& coefficient) {
        if (symbol.is(Sym::Type::Negation)) {
            const Sym::Symbol& base = symbol.as<Sym::Negation>().arg();
            if (Sym::Mul<Sym::Num, Sym::Any>::match(base)) {
                coefficient = -base.as<Sym::Product>().arg1().as<Sym::NumericConstant>().value;
                return base.as<Sym::Product>().arg2();
            }
            coefficient = -1;
            return base;
        }
        if (Sym::Mul<Sym::Num, Sym::Any>::match(symbol)) {
            coefficient = symbol.as<Sym::Product>().arg1().as<Sym::NumericConstant>().value;
            const Sym::Symbol& base = symbol.as<Sym::Product>().arg2();
            if (base.is(Sym::Type::Negation)) {
                coefficient = -coefficient;
                return base.as<Sym::Negation>().arg();
            }
            return base;
        }
        coefficient = 1;
        return symbol;
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

    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(Negation)
    DEFINE_SIMPLE_ONE_ARGUMENT_OP_ARE_EQUAL(Negation)
    DEFINE_IDENTICAL_COMPARE_TO(Negation)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Negation)
    DEFINE_SIMPLE_ONE_ARGUMENT_IS_FUNCTION_OF(Negation)

    DEFINE_INSERT_REVERSED_DERIVATIVE_AT(Negation) {
        if ((destination - 1)->is(0)) {
            return 0;
        }
        return Neg<None>::init_reverse(*destination);
    }

    DEFINE_SIMPLIFY_IN_PLACE(Negation) {
        if (arg().is(Type::Negation)) {
            arg().as<Negation>().arg().move_to(symbol());
            return true;
        }

        if (arg().is(Type::NumericConstant)) {
            symbol().init_from(NumericConstant::with_value(-arg().as<NumericConstant>().value));
            return true;
        }

        if (arg().is(Type::Addition)) {
            const size_t term_count = arg().as<Addition>().tree_size();
            if (size < arg().size() + term_count) {
                additional_required_size = term_count - 1;
                return false;
            }

            From<Addition>::Create<Addition>::WithMap<Neg>::init(
                *help_space, {{arg().as<Addition>(), term_count}});
            help_space->copy_to(symbol());

            // created addition needs to be simplified again, but without additional
            // size
            return false;
        }
        return true;
    }

    std::string Addition::to_string() const {
        return fmt::format("({}+{})", arg1().to_string(), arg2().to_string());
    }

    std::string Addition::to_tex() const {
        if (arg2().is(Type::Negation)) {
            return fmt::format("{}{}", arg1().to_tex(), arg2().as<Negation>().to_tex());
        }

        if (arg2().is(Type::NumericConstant) && arg2().as<NumericConstant>().value < 0.0) {
            return fmt::format("{}-{}", arg1().to_tex(), -arg2().as<NumericConstant>().value);
        }

        return fmt::format("{}+{}", arg1().to_tex(), arg2().to_tex());
    }

    std::string Negation::to_string() const { return fmt::format("-({})", arg().to_string()); }

    std::string Negation::to_tex() const {
        if (arg().is(Type::Addition) || arg().is(Type::Negation)) {
            return fmt::format("-\\left({}\\right)", arg().to_tex());
        }
        return fmt::format("-{}", arg().to_tex());
    }

    std::vector<Symbol> operator+(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs) {
        std::vector<Symbol> res(lhs.size() + rhs.size() + 1);
        Addition::create(lhs.data(), rhs.data(), res.data());
        return res;
    }

    std::vector<Symbol> operator-(const std::vector<Symbol>& arg) {
        std::vector<Symbol> res(arg.size() + 1);
        Negation::create(arg.data(), res.data());
        return res;
    }

    std::vector<Symbol> operator-(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs) {
        return lhs + (-rhs);
    }
}
