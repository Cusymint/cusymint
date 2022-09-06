#include "Power.cuh"

#include "Symbol.cuh"

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

    std::string Power::to_string() const {
        return "(" + arg1().to_string() + "^" + arg2().to_string() + ")";
    }

    std::vector<Symbol> operator^(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs) {
        std::vector<Symbol> res(lhs.size() + rhs.size() + 1);
        Power::create(lhs.data(), rhs.data(), res.data());
        return res;
    }
}
