#include "Symbol/Addition.cuh"
#include "Symbol/Constants.cuh"
#include "Symbol/Macros.cuh"
#include "Symbol/Product.cuh"

#include <fmt/core.h>

#include "Symbol/MetaOperators.cuh"
#include "Symbol/Symbol.cuh"
#include "Symbol/SymbolType.cuh"
#include "Symbol/TreeIterator.cuh"

namespace {
    std::string fraction_to_tex(const Sym::Symbol& numerator, const Sym::Symbol& denominator) {
        if (numerator.is(Sym::Type::Logarithm) && denominator.is(Sym::Type::Logarithm)) {
            return fmt::format(R"(\log_{{ {} }}\left({}\right))",
                               denominator.logarithm.arg().to_tex(),
                               numerator.logarithm.arg().to_tex());
        }
        return fmt::format(R"(\frac{{ {} }}{{ {} }})", numerator.to_tex(), denominator.to_tex());
    }
}

namespace Sym {
    DEFINE_TWO_ARGUMENT_COMMUTATIVE_OP_FUNCTIONS(Product)
    DEFINE_SIMPLE_TWO_ARGUMENT_OP_ARE_EQUAL(Product)
    DEFINE_IDENTICAL_COMPARE_TO(Product)
    DEFINE_TWO_ARGUMENT_OP_COMPRESS_REVERSE_TO(Product)

    DEFINE_SIMPLIFY_IN_PLACE(Product) {
        simplify_structure(help_space);
        simplify_pairs();

        if (arg1().is(0) || arg2().is(0)) {
            symbol()->init_from(NumericConstant::with_value(0));
            return true;
        }

        eliminate_ones();

        if (type == Type::Product) {
            if (!try_dividing_polynomials(help_space)) {
                return false;
            }
        }

        simplify_structure(help_space);

        return true;
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
            Symbol::move_symbol_sequence(
                rev_arg2, rev_arg2 + 1,
                d_arg1_size); // move derivative of arg1() one index back
            return Mul<Copy, None>::init_reverse(*(destination - 1), {arg2()}) - 1;
        }
        // General case: (expr2') (expr1) * (expr1') (expr2) * +
        Symbol* const second_term_dst = rev_arg2 + arg1().size() + 2;
        Symbol::move_symbol_sequence(second_term_dst, rev_arg2 + 1, d_arg1_size); // copy (expr1')
        return Add<Mul<Copy, Skip>, Mul<Copy, None>>::init_reverse(*(rev_arg2 + 1), {arg2(), d_arg1_size, arg1()}) - d_arg1_size;
    }

    __host__ __device__ bool Product::try_dividing_polynomials(Symbol* const help_space) {
        Symbol* numerator = nullptr;
        Symbol* denominator = nullptr;
        if (!arg1().is(Type::Reciprocal) && arg2().is(Type::Reciprocal)) {
            numerator = &arg1();
            denominator = &arg2().reciprocal.arg();
        }
        else if (!arg2().is(Type::Reciprocal) && arg1().is(Type::Reciprocal)) {
            numerator = &arg2();
            denominator = &arg1().reciprocal.arg();
        }
        if (numerator == nullptr || denominator == nullptr) {
            return true;
        }
        const auto optional_rank1 = numerator->is_polynomial(help_space);
        const auto optional_rank2 = denominator->is_polynomial(help_space);

        if (!optional_rank1.has_value() || optional_rank1.value() == 0 ||
            !optional_rank2.has_value() || optional_rank2.value() == 0 ||
            optional_rank1.value() <= optional_rank2.value()) {
            return true;
        }

        const auto rank1 = optional_rank1.value();
        const auto rank2 = optional_rank2.value();

        const size_t size_for_simplified_expression =
            3 + Polynomial::expanded_size_from_rank(rank1 - rank2) +
            Polynomial::expanded_size_from_rank(rank2 - 1) +
            Polynomial::expanded_size_from_rank(rank2);

        if (size < size_for_simplified_expression) {
            additional_required_size = size_for_simplified_expression - 1;
            return false;
        }

        // here we start dividing polynomials
        auto* const poly1 = help_space;
        Polynomial::make_polynomial_at(numerator, poly1);

        auto* const poly2 = poly1 + poly1->size();
        Polynomial::make_polynomial_at(denominator, poly2);

        auto* const result = (poly2 + poly2->size()) << Polynomial::with_rank(rank1 - rank2);
        Polynomial::divide_polynomials(poly1->as<Polynomial>(), poly2->as<Polynomial>(), *result);

        if (poly1->as<Polynomial>().is_zero()) {
            result->expand_to(symbol());
            return false; // maybe additional simplify required
        }

        Addition* const plus = symbol() << Addition::builder();
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

        return false; // maybe additional simplify required
    }

    DEFINE_IS_FUNCTION_OF(Product) {
        for (size_t i = 0; i < expression_count; ++i) {
            if (!expressions[i]->is(Type::Product)) {
                continue;
            }

            const auto& product_expression = expressions[i]->as<Product>();

            // TODO: In the future, this should look for correspondences in the product tree (See
            // the comment in the same function for Power symbol).
            if (arg1() == product_expression.arg1() && arg2() == product_expression.arg2()) {
                return true;
            }
        }

        return arg1().is_function_of(expressions, expression_count) &&
               arg2().is_function_of(expressions, expression_count);
    }

    __host__ __device__ bool Product::are_inverse_of_eachother(const Symbol* const expr1,
                                                               const Symbol* const expr2) {
        using Matcher = PatternPair<Inv<Same>, Same>;
        return Matcher::match_pair(*expr1, *expr2) || Matcher::match_pair(*expr2, *expr1);
    }

    DEFINE_TRY_FUSE_SYMBOLS(Product) {
        // Sprawdzenie, czy jeden z argumentów nie jest rónwy jeden jest wymagane by nie wpaść w
        // nieskończoną pętlę, jedynki i tak są potem usuwane w `eliminate_ones`.
        if (expr1->is(Type::NumericConstant) && expr2->is(Type::NumericConstant) &&
            expr1->as<NumericConstant>().value != 1.0 &&
            expr2->as<NumericConstant>().value != 1.0) {
            expr1->as<NumericConstant>().value *= expr2->as<NumericConstant>().value;
            expr2->as<NumericConstant>().value = 1.0;
            return true;
        }

        if (are_inverse_of_eachother(expr1, expr2)) {
            expr1->init_from(NumericConstant::with_value(1.0));
            expr2->init_from(NumericConstant::with_value(1.0));
            return true;
        }

        // TODO: Jakieś tożsamości trygonometryczne
        // TODO: Mnożenie potęg o tych samych podstawach

        return false;
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
        return fmt::format(R"(\frac{{1}}{{ {} }})", arg().to_string());
    }

    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(Reciprocal)
    DEFINE_SIMPLE_ONE_ARGUMENT_OP_ARE_EQUAL(Reciprocal)
    DEFINE_IDENTICAL_COMPARE_TO(Reciprocal)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Reciprocal)
    DEFINE_SIMPLE_ONE_ARGUMENT_IS_FUNCTION_OF(Reciprocal)

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

    DEFINE_INSERT_REVERSED_DERIVATIVE_AT(Reciprocal) {
        if ((destination - 1)->is(0)) {
            return 0;
        }
        return Mul<Neg<Inv<Pow<Copy, Integer<2>>>>, None>::init_reverse(*destination, {arg()});
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
