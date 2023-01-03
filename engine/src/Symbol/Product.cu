#include "Symbol/Product.cuh"

#include <fmt/core.h>
#include <fmt/format.h>
#include <list>
#include <string>

#include "Evaluation/StaticFunctions.cuh"
#include "Symbol/Addition.cuh"
#include "Symbol/Constants.cuh"
#include "Symbol/Macros.cuh"
#include "Symbol/MetaOperators.cuh"
#include "Symbol/ReverseTreeIterator.cuh"
#include "Symbol/SimplificationResult.cuh"
#include "Symbol/Symbol.cuh"
#include "Symbol/SymbolType.cuh"
#include "Symbol/TreeIterator.cuh"
#include "Utils/Order.cuh"

namespace Sym {
    DEFINE_TWO_ARGUMENT_COMMUTATIVE_OP_FUNCTIONS(Product)
    DEFINE_SIMPLE_TWO_ARGUMENT_OP_ARE_EQUAL(Product)
    DEFINE_IDENTICAL_COMPARE_TO(Product)
    DEFINE_TWO_ARGUMENT_OP_COMPRESS_REVERSE_TO(Product)

    DEFINE_SIMPLIFY_IN_PLACE(Product) {
        simplify_structure(help_space);

        if (!symbol().is(Type::Product)) {
            return true;
        }

        auto result = simplify_pairs(help_space);

        if (arg1().is(0) || arg2().is(0)) {
            symbol().init_from(NumericConstant::with_value(0));
            return true;
        }

        eliminate_ones();

        simplify_structure(help_space);
        return !is_another_loop_required(result);
    }

    DEFINE_INSERT_REVERSED_DERIVATIVE_AT(Product) {
        // Multiplication by constant
        const size_t d_arg1_size = (&destination - 1)->size();
        Symbol* const rev_arg2 = &destination - 1 - d_arg1_size;
        if ((&destination - 1)->is(0)) { // arg1() is constant
            if (rev_arg2->is(0)) {       // arg2() is constant
                return -1;
            }
            return Mul<Copy, None>::init_reverse(*(&destination - 1), {arg1()}) - 1;
        }
        if (rev_arg2->is(0)) { // arg2() is constant
            Symbol::move_symbol_sequence(rev_arg2, rev_arg2 + 1,
                                         d_arg1_size); // move derivative of arg1() one index back
            return Mul<Copy, None>::init_reverse(*(&destination - 1), {arg2()}) - 1;
        }
        // General case: (expr2') (expr1) * (expr1') (expr2) * +
        Symbol* const second_term_dst = rev_arg2 + arg1().size() + 2;
        Symbol::move_symbol_sequence(second_term_dst, rev_arg2 + 1, d_arg1_size); // copy (expr1')
        return Add<Mul<Copy, Skip>, Mul<Copy, None>>::init_reverse(*(rev_arg2 + 1),
                                                                   {arg2(), d_arg1_size, arg1()}) -
               d_arg1_size;
    }

    DEFINE_DERIVATIVE_SIZE(Product) {
        // Multiplication by constant
        const size_t d_arg1_size = (&destination - 1)->size();
        const Symbol* const rev_arg2 = &destination - 1 - d_arg1_size;
        if ((&destination - 1)->is(0)) { // arg1() is constant
            if (rev_arg2->is(0)) {       // arg2() is constant
                return -1;
            }
            return Mul<Copy, None>::size_with({arg1()}) - 1;
        }
        if (rev_arg2->is(0)) { // arg2() is constant
            return Mul<Copy, None>::size_with({arg2()}) - 1;
        }
        // General case: (expr2') (expr1) * (expr1') (expr2) * +
        return Add<Mul<Copy, Skip>, Mul<Copy, None>>::size_with({arg2(), d_arg1_size, arg1()}) -
               d_arg1_size;
    }

    __host__ __device__ double Product::exponent_coefficient(const Sym::Symbol& symbol) {
        if (!symbol.is(Type::Power)) {
            return 1.0;
        }

        return Addition::coefficient(symbol.as<Power>().arg2());
    }

    __host__ __device__ const Symbol& Product::base(const Symbol& symbol) {
        if (!symbol.is(Type::Power)) {
            return symbol;
        }

        return symbol.as<Power>().arg1();
    }

    __host__ __device__ const Symbol& Product::exponent(const Symbol& symbol) {
#ifndef __CUDA_ARCH__
        static const NumericConstant one = NumericConstant::with_value(1);
#endif

        if (!symbol.is(Type::Power)) {
#ifdef __CUDA_ARCH__
            return Static::one();
#else
            return one.symbol();
#endif
        }

        return symbol.as<Power>().arg2();
    }

    __host__ __device__ SimplificationResult Product::try_dividing_polynomials(
        Symbol* const expr1, Symbol* const expr2, Symbol* const help_space) {
        Symbol* numerator = nullptr;
        Symbol* denominator = nullptr;

        if ((!expr1->is(Type::Power) || !expr1->as<Power>().arg2().is(-1)) &&
            expr2->is(Type::Power) && expr2->as<Power>().arg2().is(-1)) {
            numerator = expr1;
            denominator = &expr2->as<Power>().arg1();
        }
        else if ((!expr2->is(Type::Power) || !expr2->as<Power>().arg2().is(-1)) &&
                 expr1->is(Type::Power) && expr1->as<Power>().arg2().is(-1)) {
            numerator = expr2;
            denominator = &expr1->as<Power>().arg1();
        }
        else {
            return SimplificationResult::NoAction;
        }

        const auto optional_rank1 = numerator->is_polynomial(help_space);
        const auto optional_rank2 = denominator->is_polynomial(help_space);

        if (!optional_rank1.has_value() || optional_rank1.value() == 0 ||
            !optional_rank2.has_value() || optional_rank2.value() == 0 ||
            optional_rank1.value() < optional_rank2.value()) {
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

        Power* const inv = &prod->arg2() << Power::builder();
        poly2->as<Polynomial>().expand_to(&inv->arg1());
        inv->seal_arg1();
        inv->arg2().init_from(NumericConstant::with_value(-1));
        inv->seal();

        prod->seal();
        plus->seal();

        return SimplificationResult::NeedsSimplification; // maybe additional simplify required
    }

    template <typename T> using LeftMul = Mul<T, Copy>;
    template <typename T> using RightMul = Mul<Copy, T>;
    __host__ __device__ SimplificationResult Product::try_split_into_sum(Symbol* const expr1,
                                                                         Symbol* const expr2,
                                                                         Symbol* const help_space) {
        Addition* sum = nullptr;
        Symbol* second_arg = nullptr;

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

        From<Addition>::Create<Addition>::WithMap<LeftMul>::init(*help_space,
                                                                 {{*sum, count}, *second_arg});
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
        if (expr1.is(Type::Power) && expr2.is(Type::Power)) {
            return Symbol::are_expressions_equal(expr1.as<Power>().arg1(),
                                                 expr2.as<Power>().arg1()) &&
                   Addition::are_equal_of_opposite_sign(expr1.as<Power>().arg2(),
                                                        expr2.as<Power>().arg2());
        }

        using Matcher = PatternPair<Inv<Same>, Same>;
        using TrigMatcher = PatternPair<Tan<Same>, Cot<Same>>;
        return Matcher::match_pair(expr1, expr2) || Matcher::match_pair(expr2, expr1) ||
               TrigMatcher::match_pair(expr1, expr2) || TrigMatcher::match_pair(expr2, expr1);
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

        // TODO: Some trigonometric identities

        return SimplificationResult::NoAction;
    }

    DEFINE_COMPARE_AND_TRY_FUSE_SYMBOLS(Product) {
        const Symbol& base1 = base(*expr1);
        const Symbol& base2 = base(*expr2);

        const auto base_order = Symbol::compare_expressions(base1, base2, *destination);

        if (base_order != Util::Order::Equal) {
            return base_order;
        }

        const Symbol& exponent1 = exponent(*expr1);
        const Symbol& exponent2 = exponent(*expr2);
        const auto exponent_order =
            Addition::compare_except_for_constant(exponent1, exponent2, *destination);

        if (exponent_order != Util::Order::Equal) {
            return exponent_order;
        }

        if constexpr (COMPARE_ONLY) {
            return exponent_order;
        }

        const double coeff1 = Addition::coefficient(exponent1);
        const double coeff2 = Addition::coefficient(exponent2);
        const double coeff_sum = coeff1 + coeff2;

        // `base1 == base2` and `exponent1 == exponent2` (modulo `NumericConstant`), so it is
        // sufficient to only check `base1` and `exponent1`
        if (base1.is(Type::NumericConstant) && exponent1.is(Type::NumericConstant)) {
            destination->init_from(
                NumericConstant::with_value(pow(base1.as<NumericConstant>().value, coeff_sum)));
        }
        else if (coeff_sum == 0.0) {
            destination->init_from(NumericConstant::with_value(1.0));
        }
        else {
            using PowCopy = Pow<Copy, None>;
            const PowCopy::AdditionalArgs args = {base1};
            PowCopy::init(*destination, args);
            destination->size() = BUILDER_SIZE;

            if (coeff_sum == 1) {
                Addition::copy_without_coefficient(destination[PowCopy::size_with(args)],
                                                   exponent1);
            }
            else {
                Addition::copy_with_coefficient(destination[PowCopy::size_with(args)], exponent1,
                                                coeff_sum);
            }

            destination->as<Power>().seal();
        }

        return Util::Order::Equal;
    }

    std::string Product::to_string() const {
        return fmt::format("({}*{})", arg1().to_string(), arg2().to_string());
    }

    namespace {
        std::string get_numerator_result(const std::list<const Symbol*>& numerator_symbols) {
            if (numerator_symbols.empty()) {
                return "1";
            }
            if (numerator_symbols.size() == 1) {
                return numerator_symbols.front()->to_tex();
            }

            std::string numerator_result;
            auto it = numerator_symbols.cbegin();
            if ((*it)->is(Type::Addition)) {
                numerator_result = fmt::format(R"(\left({}\right))", (*it)->to_tex());
            }
            else {
                numerator_result = (*it)->to_tex();
            }

            for (++it; it != numerator_symbols.cend(); ++it) {
                if ((*it)->is(Type::NumericConstant) ||
                    ((*it)->is(Type::Power) &&
                     (*it)->as<Power>().arg1().is(Type::NumericConstant)) ||
                    ((*it)->is(Type::Product) &&
                     (*it)->as<Product>().arg1().is(Type::NumericConstant))) {
                    numerator_result += " \\cdot ";
                }
                if ((*it)->is(Type::Addition) || (*it)->is_negated()) {
                    numerator_result += fmt::format(R"(\left({}\right))", (*it)->to_tex());
                }
                else {
                    numerator_result += (*it)->to_tex();
                }
            }

            return numerator_result;
        }

        std::string get_denominator_result(const std::list<const Power*>& denominator_symbols) {
            if (denominator_symbols.size() == 1) {
                return denominator_symbols.front()->to_tex_without_negation(false);
            }

            auto it = denominator_symbols.cbegin();
            std::string denominator_result = (*it)->to_tex_without_negation();

            for (++it; it != denominator_symbols.cend(); ++it) {
                if ((*it)->arg1().is(Type::NumericConstant) ||
                    ((*it)->arg1().is(Type::Power) &&
                     (*it)->arg1().as<Power>().arg1().is(Type::NumericConstant)) ||
                    ((*it)->arg1().is(Type::Product) &&
                     (*it)->arg1().as<Product>().arg1().is(Type::NumericConstant))) {
                    denominator_result += " \\cdot ";
                }
                denominator_result += (*it)->to_tex_without_negation();
            }

            return denominator_result;
        }
    }

    std::string Product::to_tex_without_negation() const {
        NumericConstant first_value = NumericConstant::with_value(1);
        std::list<const Symbol*> numerator_symbols;
        std::list<const Power*> denominator_symbols;

        ConstReverseTreeIterator<Product> iterator(this);

        if (iterator.current()->is(Type::NumericConstant)) {
            first_value.value = ::abs(iterator.current()->as<NumericConstant>().value);
            if (first_value.value != 1) {
                numerator_symbols.push_back(&first_value.symbol());
            }
            iterator.advance();
        }

        for (; iterator.is_valid(); iterator.advance()) {
            if (iterator.current()->is(Type::Power)) {
                const auto& power = iterator.current()->as<Power>();
                if (power.arg2().is_negated()) {
                    denominator_symbols.push_back(&power);
                }
                else {
                    numerator_symbols.push_back(&power.symbol());
                }
            }
            else {
                numerator_symbols.push_back(iterator.current());
            }
        }

        std::string numerator_result = get_numerator_result(numerator_symbols);

        if (denominator_symbols.empty()) {
            return numerator_result;
        }

        std::string denominator_result = get_denominator_result(denominator_symbols);

        return fmt::format(R"(\frac{{ {} }}{{ {} }})", numerator_result, denominator_result);
    }

    std::string Product::to_tex() const {
        bool negated = false;

        const Symbol& first = last_in_tree()->arg1();

        if (first.is(Type::NumericConstant) && first.as<NumericConstant>().value < 0) {
            negated = true;
        }

        return fmt::format("{}{}", negated ? "-" : "", to_tex_without_negation());
    }

    __host__ __device__ void Product::eliminate_ones() {
        for (auto* last = last_in_tree(); last >= this;
             last = (&last->symbol() - 1)->as_ptr<Product>()) {
            if (last->arg2().is(Type::NumericConstant) &&
                last->arg2().as<NumericConstant>().value == 1.0) {
                last->arg1().move_to(last->symbol());
                continue;
            }

            if (last->arg1().is(Type::NumericConstant) &&
                last->arg1().as<NumericConstant>().value == 1.0) {
                last->arg2().move_to(last->symbol());
            }
        }
    }

    std::vector<Symbol> operator*(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs) {
        std::vector<Symbol> res(lhs.size() + rhs.size() + 1);
        Product::create(lhs.data(), rhs.data(), res.data());
        return res;
    }

    std::vector<Symbol> inv(const std::vector<Symbol>& arg) {
        std::vector<Symbol> res(Inv<Copy>::size_with({*arg.data()}));
        Inv<Copy>::init(*res.data(), {*arg.data()});
        return res;
    }

    std::vector<Symbol> operator/(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs) {
        return lhs * inv(rhs);
    }
}
