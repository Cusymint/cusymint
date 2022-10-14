#include "Symbol/Product.cuh"

#include "Symbol/Symbol.cuh"
#include "Symbol/TreeIterator.cuh"
#include <fmt/core.h>

namespace Sym {
    DEFINE_TWO_ARGUMENT_COMMUTATIVE_OP_FUNCTIONS(Product)
    DEFINE_SIMPLE_TWO_ARGUMENT_OP_COMPARE(Product)
    DEFINE_TWO_ARGUMENT_OP_COMPRESS_REVERSE_TO(Product)

    DEFINE_SIMPLIFY_IN_PLACE(Product) {
        arg1().simplify_in_place(help_space);
        arg2().simplify_in_place(help_space);

        simplify_structure(help_space);
        simplify_pairs();
        eliminate_ones();

        try_simplify_polynomials(help_space);
    }

    __host__ __device__ void Product::try_simplify_polynomials(Symbol* const help_space) {
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
            return;
        }
        int const rank1 = numerator->is_polynomial();
        int const rank2 = denominator->is_polynomial();

        if (rank1 < 1 || rank2 < 1 || rank1 < rank2) {
            return;
        }

        //numerator->addition.make_polynomial_in_place(rank1, help_space);
        //denominator->addition.make_polynomial_in_place(rank2, help_space);
        auto* poly1 = help_space;
        numerator->addition.make_polynomial_to(poly1, rank1);

        auto* poly2 = (poly1 + poly1->size());
        denominator->addition.make_polynomial_to(poly2, rank2);

        auto* result = (poly2 + poly2->size()) << Polynomial::with_rank(rank1 - rank2);
        Polynomial::divide_polynomials(poly1->as<Polynomial>(), poly2->as<Polynomial>(), *result);

        // TODO larger sizes
        if (poly1->as<Polynomial>().rank < 0 && size >= result->size) { // zero polynomial
            result->this_symbol()->copy_to(this_symbol());
            return;
        }
        if (size >= poly1->size() + poly2->size() + result->size + 3) {
            Symbol* reciprocal = result->this_symbol()+result->size;
            Reciprocal::create(poly2,reciprocal);
            Symbol* product = reciprocal+reciprocal->size();
            Product::create(poly1,reciprocal,product);
            Addition::create(result->this_symbol(),product,this_symbol());
        }
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
            return fmt::format(R"(\frac{{ {} }}{{ {} }})", arg2().to_tex(),
                               arg1().reciprocal.arg().to_tex());
        }

        if (arg2().is(Type::Reciprocal)) {
            return fmt::format(R"(\frac{{ {} }}{{ {} }})", arg1().to_tex(),
                               arg2().reciprocal.arg().to_tex());
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
            arg1().copy_to(this_symbol());
            return;
        }

        if (arg1().is(Type::NumericConstant) && arg1().numeric_constant.value == 1.0) {
            arg2().copy_to(this_symbol());
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

    DEFINE_SIMPLIFY_IN_PLACE(Reciprocal) { arg().simplify_in_place(help_space); }

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
