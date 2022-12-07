#include "Symbol/Addition.cuh"
#include "Symbol/Constants.cuh"
#include "Symbol/Macros.cuh"
#include "Symbol/Product.cuh"

#include <fmt/core.h>

#include "Symbol/MetaOperators.cuh"
#include "Symbol/SimplificationResult.cuh"
#include "Symbol/Symbol.cuh"
#include "Symbol/SymbolType.cuh"
#include "Symbol/TreeIterator.cuh"
#include "Utils/Order.cuh"

namespace {
    std::string fraction_to_tex(const Sym::Symbol& numerator, const Sym::Symbol& denominator) {
        if (numerator.is(Sym::Type::Logarithm) && denominator.is(Sym::Type::Logarithm)) {
            return fmt::format(R"(\log_{{ {} }}\left({}\right))",
                               denominator.logarithm.arg().to_tex(),
                               numerator.logarithm.arg().to_tex());
        }
        return fmt::format(R"(\frac{{ {} }}{{ {} }})", numerator.to_tex(), denominator.to_tex());
    }

    __host__ __device__ void extract_base_exponent_and_coefficient(const Sym::Symbol& symbol,
                                                                   const Sym::Symbol*& base,
                                                                   const Sym::Symbol*& exponent,
                                                                   double& coefficient) {
        const Sym::Symbol* inner = &symbol;
        double reciprocal_coefficient = 1;

        if (inner->is(Sym::Type::Reciprocal)) {
            inner = &inner->as<Sym::Reciprocal>().arg();
            reciprocal_coefficient = -1;
        }

        if (!inner->is(Sym::Type::Power)) {
            base = inner;
            // we do not assign to exponent as it has default value
            coefficient = reciprocal_coefficient;
            return;
        }

        base = &inner->as<Sym::Power>().arg1();

        exponent = &Sym::Addition::extract_base_and_coefficient(inner->as<Sym::Power>().arg2(),
                                                                coefficient);

        if (base->is(Sym::Type::Reciprocal)) {
            base = &base->as<Sym::Reciprocal>().arg();
            coefficient = -coefficient;
        }
        coefficient *= reciprocal_coefficient;
    }
}

namespace Sym {
    DEFINE_TWO_ARGUMENT_COMMUTATIVE_OP_FUNCTIONS(Product)
    DEFINE_SIMPLE_TWO_ARGUMENT_OP_ARE_EQUAL(Product)
    DEFINE_IDENTICAL_COMPARE_TO(Product)
    DEFINE_TWO_ARGUMENT_OP_COMPRESS_REVERSE_TO(Product)

    DEFINE_SIMPLIFY_IN_PLACE(Product) {
        simplify_structure(help_space);

        if (!symbol()->is(Type::Product)) {
            return true;
        }

        auto result = simplify_pairs(help_space);

        if (arg1().is(0) || arg2().is(0)) {
            symbol()->init_from(NumericConstant::with_value(0));
            return true;
        }

        eliminate_ones();

        simplify_structure(help_space);
        return !is_another_loop_required(result);
    }

    DEFINE_INSERT_REVERSED_DERIVATIVE_AT(Product) {
        // Multiplication by constant
        const size_t d_arg1_size = (destination - 1)->size();
        Symbol* const rev_arg2 = destination - 1 - d_arg1_size;
        if ((destination - 1)->is(0)) { // arg1() is constant
            if (rev_arg2->is(0)) {      // arg2() is constant
                return -1;
            }
            return Mul<Copy, None>::init_reverse(*(destination - 1), {arg1()}) - 1;
        }
        if (rev_arg2->is(0)) { // arg2() is constant
            Symbol::move_symbol_sequence(rev_arg2, rev_arg2 + 1,
                                         d_arg1_size); // move derivative of arg1() one index back
            return Mul<Copy, None>::init_reverse(*(destination - 1), {arg2()}) - 1;
        }
        // General case: (expr2') (expr1) * (expr1') (expr2) * +
        Symbol* const second_term_dst = rev_arg2 + arg1().size() + 2;
        Symbol::move_symbol_sequence(second_term_dst, rev_arg2 + 1, d_arg1_size); // copy (expr1')
        return Add<Mul<Copy, Skip>, Mul<Copy, None>>::init_reverse(*(rev_arg2 + 1),
                                                                   {arg2(), d_arg1_size, arg1()}) -
               d_arg1_size;
    }

    __host__ __device__ SimplificationResult Product::try_dividing_polynomials(
        Symbol* const expr1, Symbol* const expr2, Symbol* const help_space) {
        Symbol* numerator = nullptr;
        Symbol* denominator = nullptr;
        if (!expr1->is(Type::Reciprocal) && expr2->is(Type::Reciprocal)) {
            numerator = expr1;
            denominator = &expr2->as<Reciprocal>().arg();
        }
        else if (!expr2->is(Type::Reciprocal) && expr1->is(Type::Reciprocal)) {
            numerator = expr2;
            denominator = &expr1->as<Reciprocal>().arg();
        }
        if (numerator == nullptr || denominator == nullptr) {
            return SimplificationResult::NoAction;
        }

        const auto optional_rank1 = numerator->is_polynomial(help_space);
        const auto optional_rank2 = denominator->is_polynomial(help_space);

        if (!optional_rank1.has_value() || optional_rank1.value() == 0 ||
            !optional_rank2.has_value() || optional_rank2.value() == 0 ||
            optional_rank1.value() <= optional_rank2.value()) {
            return SimplificationResult::NoAction;
        }

        const auto rank1 = optional_rank1.value();
        const auto rank2 = optional_rank2.value();

        const size_t size_for_simplified_expression =
            3 + Polynomial::expanded_size_from_rank(rank1 - rank2) +
            Polynomial::expanded_size_from_rank(rank2 - 1) +
            Polynomial::expanded_size_from_rank(rank2);

        Symbol* const longer_expr = expr1->size() > expr2->size() ? expr1 : expr2;
        Symbol* const shorter_expr = longer_expr == expr1 ? expr2 : expr1;

        if (longer_expr->size() < size_for_simplified_expression) {
            longer_expr->additional_required_size() = size_for_simplified_expression - 1;
            return SimplificationResult::NeedsSpace;
        }

        // here we start dividing polynomials
        auto* const poly1 = help_space;
        Polynomial::make_polynomial_at(numerator, poly1);

        auto* const poly2 = poly1 + poly1->size();
        Polynomial::make_polynomial_at(denominator, poly2);

        auto* const result = (poly2 + poly2->size()) << Polynomial::with_rank(rank1 - rank2);
        Polynomial::divide_polynomials(poly1->as<Polynomial>(), poly2->as<Polynomial>(), *result);

        shorter_expr->init_from(NumericConstant::with_value(1.0));

        if (poly1->as<Polynomial>().is_zero()) {
            result->expand_to(longer_expr);
            return SimplificationResult::NeedsSimplification; // maybe additional simplify
                                                              // required
        }

        Addition* const plus = longer_expr << Addition::builder();
        result->expand_to(&plus->arg1());
        plus->seal_arg1();

        Product* const prod = &plus->arg2() << Product::builder();
        poly1->as<Polynomial>().expand_to(&prod->arg1());
        prod->seal_arg1();

        Reciprocal* const rec = &prod->arg2() << Reciprocal::builder();
        poly2->as<Polynomial>().expand_to(&rec->arg());
        rec->seal();

        prod->seal();
        plus->seal();

        return SimplificationResult::NeedsSimplification; // maybe additional simplify required
    }

    template <typename T> using LeftMul = Mul<T, Copy>;
    template <typename T> using RightMul = Mul<Copy, T>;
     __host__ __device__ SimplificationResult Product::try_split_into_sum(Symbol* const expr1, Symbol* const expr2, Symbol* const help_space) {
        Addition* sum;
        Symbol* second_arg;

         if (!expr1->is(Type::Addition) && !expr2->is(Type::Addition)) {
            return SimplificationResult::NoAction;
        }

        if (expr1->is(Type::Addition)) {
            sum = expr1->as_ptr<Addition>();
            second_arg = expr2;
        }
        else {
            sum = expr2->as_ptr<Addition>();
            second_arg = expr1;
        }

        const auto count = sum->tree_size();
        const auto actual_size = sum->arg1().size() + sum->arg2().size() + 1;
        if (sum->size < count * (second_arg->size() + 1) + actual_size) {
            sum->additional_required_size = count * (second_arg->size() + 1);
            return SimplificationResult::NeedsSpace;
        }

        From<Addition>::Create<Addition>::WithMap<LeftMul>::init(*help_space, {{*sum, count}, *second_arg});
        second_arg->init_from(NumericConstant::with_value(1));

        help_space->copy_to(sum->symbol());
        return SimplificationResult::NeedsSimplification;
    }

    DEFINE_IS_FUNCTION_OF(Product) {
        BASE_TWO_ARGUMENT_IS_FUNCTION_OF

        for (size_t i = 0; i < expression_count; ++i) {
            if (!expressions[i]->is(Type::Product)) {
                continue;
            }

            const auto& product_expression = expressions[i]->as<Product>();

            // TODO: In the future, this should look for correspondences in the product tree (See
            // the comment in the same function for Power symbol).
            if (Symbol::are_expressions_equal(arg1(), product_expression.arg1()) &&
                Symbol::are_expressions_equal(arg2(), product_expression.arg2())) {
                return true;
            }
        }

        return false;
    }

    __host__ __device__ bool Product::are_inverse_of_eachother(const Symbol& expr1,
                                                               const Symbol& expr2) {
        using Matcher = PatternPair<Inv<Same>, Same>;
        return Matcher::match_pair(expr1, expr2) || Matcher::match_pair(expr2, expr1);
    }

    DEFINE_TRY_FUSE_SYMBOLS(Product) {
        // Sprawdzenie, czy jeden z argumentów nie jest rónwy jeden jest wymagane by nie wpaść w
        // nieskończoną pętlę, jedynki i tak są potem usuwane w `eliminate_ones`.
        if (expr1->is(Type::NumericConstant) && expr2->is(Type::NumericConstant) &&
            expr1->as<NumericConstant>().value != 1.0 &&
            expr2->as<NumericConstant>().value != 1.0) {
            expr1->as<NumericConstant>().value *= expr2->as<NumericConstant>().value;
            expr2->as<NumericConstant>().value = 1.0;
            return SimplificationResult::Success;
        }

        if (are_inverse_of_eachother(*expr1, *expr2)) {
            expr1->init_from(NumericConstant::with_value(1.0));
            expr2->init_from(NumericConstant::with_value(1.0));
            return SimplificationResult::Success;
        }

        const SimplificationResult divide_polynomials_result =
            try_dividing_polynomials(expr1, expr2, help_space);
        if (divide_polynomials_result != SimplificationResult::NoAction) {
            return divide_polynomials_result;
        }

        const auto split_result = try_split_into_sum(expr1, expr2, help_space);
        if (split_result != SimplificationResult::NoAction) {
            return split_result;
        }

        // TODO: Jakieś tożsamości trygonometryczne

        return SimplificationResult::NoAction;
    }

    DEFINE_COMPARE_AND_TRY_FUSE_SYMBOLS(Product) {
        NumericConstant one = NumericConstant::with_value(1);

        const Symbol* base1;
        const Symbol* base2;
        const Symbol* exponent1 = one.symbol();
        const Symbol* exponent2 = one.symbol();
        double coef1;
        double coef2;

        extract_base_exponent_and_coefficient(*expr1, base1, exponent1, coef1);
        extract_base_exponent_and_coefficient(*expr2, base2, exponent2, coef2);

        const auto base_order = Symbol::compare_expressions(*base1, *base2, *destination);

        if (base_order != Util::Order::Equal) {
            return base_order;
        }

        if (exponent1->is(Type::NumericConstant) && exponent2->is(Type::NumericConstant)) {
            const double exp_sum = coef1 * exponent1->as<NumericConstant>().value +
                                   coef2 * exponent2->as<NumericConstant>().value;
            if (exp_sum == 0) {
                destination->init_from(NumericConstant::with_value(1));
                return Util::Order::Equal;
            }
            if (exp_sum == 1) {
                base1->copy_to(destination);
                return Util::Order::Equal;
            }
            if (base1->is(Type::NumericConstant)) {
                destination->init_from(
                    NumericConstant::with_value(pow(base1->as<NumericConstant>().value, exp_sum)));
                return Util::Order::Equal;
            }
            if (exp_sum == -1) {
                Inv<Copy>::init(*destination, {*base1});
                return Util::Order::Equal;
            }
            
            Pow<Copy, Num>::init(*destination, {*base1, exp_sum});
            return Util::Order::Equal;
        }

        const auto order = Symbol::compare_expressions(*exponent1, *exponent2, *destination);

        if constexpr (COMPARE_ONLY) {
            return order;
        }

        if (order != Util::Order::Equal) {
            return order;
        }

        const double sum = coef1 + coef2;
        if (sum == 0) {
            destination->init_from(one);
        }
        else if (sum == 1) {
            Pow<Copy, Copy>::init(*destination, {*base1, *exponent1});
        }
        else if (sum == -1) {
            Pow<Inv<Copy>, Copy>::init(*destination, {*base1, *exponent1});
        }
        else {
            Pow<Copy, Mul<Num, Copy>>::init(*destination, {*base1, sum, *exponent1});
        }
        return Util::Order::Equal;
    }

    std::string Product::to_string() const {
        return fmt::format("({}*{})", arg1().to_string(), arg2().to_string());
    }

    std::string Product::to_tex() const {
        if (arg1().is(Type::Reciprocal)) {
            return fraction_to_tex(arg2(), arg1().reciprocal.arg());
        }

        if (arg2().is(Type::Reciprocal)) {
            return fraction_to_tex(arg1(), arg2().reciprocal.arg());
        }

        std::string arg1_pattern = "{}";
        std::string arg2_pattern = "{}";
        std::string cdot = " ";
        if (arg1().is(Type::Addition) || arg1().is(Type::Negation)) {
            arg1_pattern = R"(\left({}\right))";
        }
        if (arg2().is(Type::Addition) || arg2().is(Type::Negation)) {
            arg2_pattern = R"(\left({}\right))";
        }
        if (arg2().is(Type::Negation) || arg2().is(Type::NumericConstant)) {
            cdot = " \\cdot ";
        }
        return fmt::format(arg1_pattern + cdot + arg2_pattern, arg1().to_tex(), arg2().to_tex());
    }

    __host__ __device__ void Product::eliminate_ones() {
        for (auto* last = last_in_tree(); last >= this;
             last = (last->symbol() - 1)->as_ptr<Product>()) {
            if (last->arg2().is(Type::NumericConstant) &&
                last->arg2().numeric_constant.value == 1.0) {
                last->arg1().move_to(last->symbol());
                continue;
            }

            if (last->arg1().is(Type::NumericConstant) &&
                last->arg1().numeric_constant.value == 1.0) {
                last->arg2().move_to(last->symbol());
            }
        }
    }

    std::string Reciprocal::to_string() const { return fmt::format("(1/{})", arg().to_string()); }

    std::string Reciprocal::to_tex() const {
        return fmt::format(R"(\frac{{1}}{{ {} }})", arg().to_tex());
    }

    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(Reciprocal)
    DEFINE_SIMPLE_ONE_ARGUMENT_OP_ARE_EQUAL(Reciprocal)
    DEFINE_IDENTICAL_COMPARE_TO(Reciprocal)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Reciprocal)
    DEFINE_SIMPLE_ONE_ARGUMENT_IS_FUNCTION_OF(Reciprocal)
    DEFINE_ONE_ARG_OP_DERIVATIVE(Reciprocal, (Neg<Inv<Pow<Copy, Integer<2>>>>))

    DEFINE_SIMPLIFY_IN_PLACE(Reciprocal) {
        if (arg().is(Type::Reciprocal)) {
            arg().as<Reciprocal>().arg().copy_to(symbol());
            return true;
        }

        if (arg().is(Type::NumericConstant)) {
            symbol()->init_from(
                NumericConstant::with_value(1.0 / arg().as<NumericConstant>().value));
            return true;
        }

        return true;
    }

    std::vector<Symbol> operator*(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs) {
        std::vector<Symbol> res(lhs.size() + rhs.size() + 1);
        Product::create(lhs.data(), rhs.data(), res.data());
        return res;
    }

    std::vector<Symbol> inv(const std::vector<Symbol>& arg) {
        std::vector<Symbol> res(arg.size() + 1);
        Inv<Copy>::init(*res.data(), {*arg.data()});
        return res;
    }

    std::vector<Symbol> operator/(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs) {
        return lhs * inv(rhs);
    }
}
