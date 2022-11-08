#include "Symbol/Addition.cuh"
#include "Symbol/Product.cuh"

#include "Symbol/Symbol.cuh"
#include "Symbol/TreeIterator.cuh"
#include <cstdio>
#include <fmt/core.h>

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
        eliminate_ones();

        return try_simplify_polynomials(help_space);;
    }

    __host__ __device__ bool Product::try_simplify_polynomials(Symbol* const help_space) {
        Symbol* numerator = nullptr;
        Symbol* denominator = nullptr;
        if (arg1().is(Type::Addition) && arg2().is(Type::Reciprocal) &&
            arg2().reciprocal.arg().is(Type::Addition)) {
            numerator = &arg1();
            denominator = &arg2().reciprocal.arg();
        }
        if (arg2().is(Type::Addition) && arg1().is(Type::Reciprocal) &&
            arg1().reciprocal.arg().is(Type::Addition)) {
            numerator = &arg2();
            denominator = &arg1().reciprocal.arg();
        }
        if (numerator == nullptr || denominator == nullptr) {
            return true;
        }
        int const rank1 = numerator->is_polynomial();
        int const rank2 = denominator->is_polynomial();

        if (rank1 < 1 || rank2 < 1 || rank1 <= rank2) {
            return true;
        }

        const size_t size_for_simplified_expression =
            3 + Polynomial::expanded_size_from_rank(rank1 - rank2) +
            Polynomial::expanded_size_from_rank(rank2 - 1) +
            Polynomial::expanded_size_from_rank(rank2);

        printf("%lu %lu\n", size, size_for_simplified_expression);

        if (size < size_for_simplified_expression) {
            additional_required_size = size_for_simplified_expression - size;
            return false;
        }

        // here we start dividing polynomials
        auto* poly1 = help_space;
        numerator->addition.make_polynomial_to(poly1, rank1);

        auto* poly2 = poly1 + poly1->size();
        denominator->addition.make_polynomial_to(poly2, rank2);

        auto* result = (poly2 + poly2->size()) << Polynomial::with_rank(rank1 - rank2);
        Polynomial::divide_polynomials(poly1->as<Polynomial>(), poly2->as<Polynomial>(), *result);

        Symbol* destination = symbol();

        if (poly1->as<Polynomial>().is_zero()) {
            result->expand_to(destination);
            return true;
        }

        Addition* plus = symbol() << Addition::builder();
        result->expand_to(&plus->arg1());
        plus->seal_arg1();

        Product* prod = &plus->arg2() << Product::builder();
        poly1->as<Polynomial>().expand_to(&prod->arg1());
        prod->seal_arg1();

        Reciprocal* rec = &prod->arg2() << Reciprocal::builder();
        poly2->as<Polynomial>().expand_to(&rec->arg());
        rec->seal();
        
        prod->seal();
        plus->seal();

        return false; // maybe additional simplify required
        // // TODO larger sizes
        // if (poly1->as<Polynomial>().rank < 0 && size >= result->size) { // zero polynomial
        //     result->symbol()->copy_to(symbol());
        //     return;
        // }
        // if (size >= poly1->size() + poly2->size() + result->size + 3) {
        //     Symbol* reciprocal = result->symbol() + result->size;
        //     Reciprocal::create(poly2, reciprocal);
        //     Symbol* product = reciprocal + reciprocal->size();
        //     Product::create(poly1, reciprocal, product);
        //     Addition::create(result->symbol(), product, symbol());
        // }
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
            expr1->numeric_constant.value != 1.0 && expr2->numeric_constant.value != 1.0) {
            expr1->numeric_constant.value *= expr2->numeric_constant.value;
            expr2->numeric_constant.value = 1.0;
            return true;
        }

        if (are_inverse_of_eachother(expr1, expr2)) {
            expr1->numeric_constant = NumericConstant::with_value(1.0);
            expr2->numeric_constant = NumericConstant::with_value(1.0);
            return true;
        }

        // TODO: Jakieś tożsamości trygonometryczne
        // TODO: Mnożenie potęg o tych samych podstawach

        return false;
    }

    std::string Product::to_string() const {
        if (arg1().is(Type::Reciprocal)) {
            return fmt::format("({}/{})", arg2().to_string(), arg1().reciprocal.arg().to_string());
        }

        if (arg2().is(Type::Reciprocal)) {
            return fmt::format("({}/{})", arg1().to_string(), arg2().reciprocal.arg().to_string());
        }

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
        if (arg1().is(Type::Product)) {
            arg1().product.eliminate_ones();
        }

        if (arg2().is(Type::NumericConstant) && arg2().numeric_constant.value == 1.0) {
            arg1().copy_to(symbol());
            return;
        }

        if (arg1().is(Type::NumericConstant) && arg1().numeric_constant.value == 1.0) {
            arg2().copy_to(symbol());
        }
    }

    __host__ __device__ int Product::is_polynomial() const {
        // Checks if product is a monomial (maybe TODO check eg. (x+2)^5*(x+1)*x^2)
        if (arg1().is(Type::Addition) || arg2().is(Type::Addition)) {
            return -1;
        }

        int const rank1 = arg1().is_polynomial();
        int const rank2 = arg2().is_polynomial();

        return rank1 < 0 || rank2 < 0 ? -1 : (rank1 + rank2);
    }

    __host__ __device__ double Product::get_monomial_coefficient() const {
        return arg1().get_monomial_coefficient() * arg2().get_monomial_coefficient();
    }

    std::string Reciprocal::to_string() const { return fmt::format("(1/{})", arg().to_string()); }

    std::string Reciprocal::to_tex() const {
        return fmt::format(R"(\frac{{1}}{{ {} }})", arg().to_string());
    }

    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(Reciprocal)
    DEFINE_SIMPLE_ONE_ARGUMETN_OP_COMPARE(Reciprocal)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Reciprocal)
    DEFINE_SIMPLE_ONE_ARGUMENT_IS_FUNCTION_OF(Reciprocal)

    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(Reciprocal)

    std::vector<Symbol> operator*(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs) {
        std::vector<Symbol> res(lhs.size() + rhs.size() + 1);
        Product::create(lhs.data(), rhs.data(), res.data());
        return res;
    }

    std::vector<Symbol> operator/(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs) {
        std::vector<Symbol> res(rhs.size() + 1);

        Reciprocal* const reciprocal = res.data() << Reciprocal::builder();
        rhs.data()->copy_to(&reciprocal->arg());
        reciprocal->seal();

        return lhs * res;
    }
}
