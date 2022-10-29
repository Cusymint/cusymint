#include "Addition.cuh"

#include "Symbol.cuh"
#include "TreeIterator.cuh"
#include "Utils/Cuda.cuh"

#include <fmt/core.h>

namespace Sym {
    DEFINE_TWO_ARGUMENT_COMMUTATIVE_OP_FUNCTIONS(Addition)
    DEFINE_SIMPLE_ONE_ARGUMETN_OP_COMPARE(Addition)
    DEFINE_TWO_ARGUMENT_OP_COMPRESS_REVERSE_TO(Addition)

    DEFINE_SIMPLIFY_IN_PLACE(Addition) {
        simplify_structure(help_space);
        simplify_pairs();
        eliminate_zeros();
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
            if (arg1() == addition_expression.arg1() && arg2() == addition_expression.arg2()) {
                return true;
            }
        }

        return arg1().is_function_of(expressions, expression_count) &&
               arg2().is_function_of(expressions, expression_count);
    }

    __host__ __device__ bool Addition::is_sine_cosine_squared_sum(const Symbol* const expr1,
                                                                  const Symbol* const expr2) {
        bool is_expr1_sine_squared = expr1->is(Type::Power) && expr1->power.arg1().is(Type::Sine) &&
                                     expr1->power.arg1().sine.arg().is(Type::Variable) &&
                                     expr1->power.arg2().is(Type::NumericConstant) &&
                                     expr1->power.arg2().numeric_constant.value == 2.0;

        bool is_expr2_cosine_squared = expr2->is(Type::Power) &&
                                       expr2->power.arg1().is(Type::Cosine) &&
                                       expr2->power.arg1().cosine.arg().is(Type::Variable) &&
                                       expr2->power.arg2().is(Type::NumericConstant) &&
                                       expr2->power.arg2().numeric_constant.value == 2.0;

        return is_expr1_sine_squared && is_expr2_cosine_squared;
    }

    __host__ __device__ bool Addition::are_equal_of_opposite_sign(const Symbol* const expr1,
                                                                  const Symbol* const expr2) {
        return expr1->is(Type::Negation) && Symbol::compare_trees(&expr1->negation.arg(), expr2) ||
               expr2->is(Type::Negation) && Symbol::compare_trees(&expr2->negation.arg(), expr1);
    }

    DEFINE_TRY_FUSE_SYMBOLS(Addition) {
        // Sprawdzenie, czy jeden z argumentów nie jest rónwy zero jest wymagane by nie wpaść w
        // nieskończoną pętlę, zera i tak są potem usuwane w `eliminate_zeros`.
        if (expr1->is(Type::NumericConstant) && expr2->is(Type::NumericConstant) &&
            expr1->numeric_constant.value != 0.0 && expr2->numeric_constant.value != 0.0) {
            expr1->numeric_constant.value += expr2->numeric_constant.value;
            expr2->numeric_constant.value = 0.0;
            return true;
        }

        if (are_equal_of_opposite_sign(expr1, expr2)) {
            expr1->numeric_constant = NumericConstant::with_value(0.0);
            expr2->numeric_constant = NumericConstant::with_value(0.0);
            return true;
        }

        // TODO: Jakieś inne tożsamości trygonometryczne
        // TODO: Rozszerzyć na kombinacje liniowe, np x*sin^2(x) + x*cos^2(x) = x
        if (is_sine_cosine_squared_sum(expr1, expr2) || is_sine_cosine_squared_sum(expr2, expr1)) {
            expr1->numeric_constant = NumericConstant::with_value(1.0);
            expr2->numeric_constant = NumericConstant::with_value(0.0);
            return true;
        }

        // TODO: Dodawanie gdy to samo jest tylko przemnożone przez stałą
        // TODO: Jedynka hiperboliczna gdy funkcje hiperboliczne będą już zaimplementowane

        return false;
    }

    __host__ __device__ void Addition::eliminate_zeros() {
        if (arg1().is(Type::Addition)) {
            arg1().addition.eliminate_zeros();
        }

        if (arg2().is(Type::NumericConstant) && arg2().numeric_constant.value == 0.0) {
            arg1().copy_to(symbol());
            return;
        }

        if (arg1().is(Type::NumericConstant) && arg1().numeric_constant.value == 0.0) {
            arg2().copy_to(symbol());
        }
    }

    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(Negation)
    DEFINE_SIMPLE_ONE_ARGUMETN_OP_COMPARE(Negation)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Negation)
    DEFINE_SIMPLE_ONE_ARGUMENT_IS_FUNCTION_OF(Negation)

    DEFINE_SIMPLIFY_IN_PLACE(Negation) {
        arg().simplify_in_place(help_space);

        if (arg().is(Type::Negation)) {
            arg().negation.arg().copy_to(symbol());
            return;
        }

        if (arg().is(Type::NumericConstant)) {
            *as<NumericConstant>() = NumericConstant::with_value(-arg().numeric_constant.value);
        }
    }

    std::string Addition::to_string() const {
        if (arg2().is(Type::Negation)) {
            return fmt::format("({}-{})", arg1().to_string(), arg2().negation.arg().to_string());
        }

        if (arg2().is(Type::NumericConstant) && arg2().numeric_constant.value < 0.0) {
            return fmt::format("({}-{})", arg1().to_string(), -arg2().numeric_constant.value);
        }

        return fmt::format("({}+{})", arg1().to_string(), arg2().to_string());
    }

    std::string Addition::to_tex() const {
        if (arg2().is(Type::Negation)) {
            return fmt::format("{}{}", arg1().to_tex(), arg2().negation.to_tex());
        }

        if (arg2().is(Type::NumericConstant) && arg2().numeric_constant.value < 0.0) {
            return fmt::format("{}-{}", arg1().to_tex(), -arg2().numeric_constant.value);
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
