#include "Power.cuh"

#include "Symbol.cuh"
#include <fmt/core.h>

namespace Sym {
    DEFINE_TWO_ARGUMENT_OP_FUNCTIONS(Power)
    DEFINE_SIMPLE_TWO_ARGUMENT_OP_COMPARE(Power)
    DEFINE_TWO_ARGUMENT_OP_COMPRESS_REVERSE_TO(Power)

    DEFINE_SIMPLIFY_IN_PLACE(Power) {
        arg1().simplify_in_place(help_space);
        arg2().simplify_in_place(help_space);

        if (arg2().is(Type::NumericConstant) && arg2().numeric_constant.value == 0.0) {
            Symbol::from(this)->numeric_constant = NumericConstant::with_value(1.0);
            return;
        }

        if (arg1().is(Type::NumericConstant) && arg2().is(Type::NumericConstant)) {
            double value1 = arg1().numeric_constant.value;
            double value2 = arg2().numeric_constant.value;
            Symbol::from(this)->numeric_constant = NumericConstant::with_value(pow(value1, value2));
            return;
        }

        // (a^b)^c -> a^(b*c)
        if (arg1().is(Type::Power)) {
            Symbol::from(this)->copy_to(help_space);
            Power* const this_copy = &help_space->power;

            *this = Power::create();
            this_copy->arg1().power.arg1().copy_to(&arg1());
            seal_arg1();

            Product* const product = &arg2() << Product::create();
            this_copy->arg1().power.arg2().copy_to(&product->arg1());
            product->seal_arg1();
            this_copy->arg2().copy_to(&product->arg2());
            product->seal();

            seal();

            return;
        }
    }

    DEFINE_IS_FUNCTION_OF(Power) {
        for (size_t i = 0; i < expression_count; ++i) {
            if (!expressions[i]->is(Type::Power)) {
                continue;
            }

            const auto& power_expression = expressions[i]->as<Power>();

            // TODO: In the future, this should look for correspondences in the product tree of
            // arg1(). For example, if `this` is `e^(pi*x*2*sin(x))`, then this function should
            // return true when `power_expression` is `e^(sin(x)*x)`.
            if (arg1() == power_expression.arg1() && arg2() == power_expression.arg2()) {
                return true;
            }
        }

        return arg1().is_function_of(expressions, expression_count) &&
               arg2().is_function_of(expressions, expression_count);
    }

    std::string Power::to_string() const {
        return fmt::format("({}^{})", arg1().to_string(), arg2().to_string());
    }

    std::string Power::to_tex() const {
        if (arg1().is(Type::Addition) || arg1().is(Type::Product) || arg1().is(Type::Negation) ||
            arg1().is(Type::Reciprocal) || arg1().is(Type::Power)) {
            return fmt::format(R"(\left({}\right)^{{ {} }})", arg1().to_tex(), arg2().to_tex());
        }
        return fmt::format("{}^{{ {} }}", arg1().to_tex(), arg2().to_tex());
    }

    std::vector<Symbol> operator^(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs) {
        std::vector<Symbol> res(lhs.size() + rhs.size() + 1);
        Power::create(lhs.data(), rhs.data(), res.data());
        return res;
    }
}
