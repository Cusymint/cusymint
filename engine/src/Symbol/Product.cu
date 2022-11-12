#include "Symbol/Product.cuh"

#include <fmt/core.h>

#include "Symbol/MetaOperators.cuh"
#include "Symbol/Symbol.cuh"
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
    DEFINE_SIMPLE_TWO_ARGUMENT_OP_COMPARE(Product)
    DEFINE_TWO_ARGUMENT_OP_COMPRESS_REVERSE_TO(Product)

    DEFINE_SIMPLIFY_IN_PLACE(Product) {
        simplify_structure(help_space);
        simplify_pairs();

        if (arg1().is(0) || arg2().is(0)) {
            symbol()->init_from(NumericConstant::with_value(0));
            return true;
        }

        eliminate_ones();

        return true;
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
        return expr1->is(Type::Reciprocal) &&
                   Symbol::compare_trees(&expr1->reciprocal.arg(), expr2) ||
               expr2->is(Type::Reciprocal) &&
                   Symbol::compare_trees(&expr2->reciprocal.arg(), expr1);
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
                last->arg1().copy_to(last->symbol());
                continue;
            }

            if (last->arg1().is(Type::NumericConstant) &&
                last->arg1().numeric_constant.value == 1.0) {
                last->arg2().copy_to(last->symbol());
            }
        }
    }

    std::string Reciprocal::to_string() const { return fmt::format("(1/{})", arg().to_string()); }

    std::string Reciprocal::to_tex() const {
        return fmt::format(R"(\frac{{1}}{{ {} }})", arg().to_string());
    }

    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(Reciprocal)
    DEFINE_SIMPLE_ONE_ARGUMETN_OP_COMPARE(Reciprocal)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Reciprocal)
    DEFINE_SIMPLE_ONE_ARGUMENT_IS_FUNCTION_OF(Reciprocal)

    DEFINE_SIMPLIFY_IN_PLACE(Reciprocal) {
        if (arg().is(Type::Reciprocal)) {
            arg().as<Reciprocal>().arg().copy_to(symbol());
            return true;
        }

        if (arg().is(1)) {
            symbol()->init_from(NumericConstant::with_value(1));
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
