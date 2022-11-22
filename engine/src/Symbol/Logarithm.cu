#include "InverseTrigonometric.cuh"

#include <fmt/core.h>

#include "Symbol.cuh"
#include "Symbol/Constants.cuh"
#include "Symbol/Macros.cuh"
#include "Symbol/MetaOperators.cuh"
#include "Symbol/Product.cuh"

namespace Sym {
    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(Logarithm)
    DEFINE_SIMPLE_ONE_ARGUMENT_OP_ARE_EQUAL(Logarithm)
    DEFINE_IDENTICAL_COMPARE_TO(Logarithm)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Logarithm)
    DEFINE_SIMPLE_ONE_ARGUMENT_IS_FUNCTION_OF(Logarithm)
    DEFINE_ONE_ARG_OP_DERIVATIVE(Logarithm, Inv<Copy>)

    DEFINE_SIMPLIFY_IN_PLACE(Logarithm) {
        if (arg().is(Type::NumericConstant) && arg().numeric_constant.value == 1) {
            // ln(1) = 0
            *(this->as<NumericConstant>()) = NumericConstant::with_value(0);
            return true;
        }
        if (arg().is(Type::KnownConstant) && arg().known_constant.value == KnownConstantValue::E) {
            // ln(e) = 1
            *(this->as<NumericConstant>()) = NumericConstant::with_value(1);
            return true;
        }
        if (arg().is(Type::Power)) {
            // Here we do the following transformation: ln(f(x)^g(x)) = ln(f(x))*g(x).
            // Structure in memory before operation: LN | ^  | f(x) | g(x)
            // Structure after operation:            *  | LN | f(x) | g(x)
            // Therefore, it is not necessary to do any copying, we only need to
            // change types of 2 first symbols, size of the second and add second_arg_offset
            // to new Product symbol.
            arg().power.type = Type::Logarithm;
            arg().as<Logarithm>().seal();
            this->type = Type::Product;
            this->as<Product>()->seal_arg1();
            return true;
        }
        return true;
    }

    [[nodiscard]] std::string Logarithm::to_string() const {
        return fmt::format("ln({})", arg().to_string());
    }

    [[nodiscard]] std::string Logarithm::to_tex() const {
        return fmt::format(R"(\ln\left({}\right))", arg().to_tex());
    }

    std::vector<Symbol> log(const std::vector<Symbol>& base, const std::vector<Symbol>& arg) {
        return ln(arg) / ln(base);
    }

    std::vector<Symbol> ln(const std::vector<Symbol>& arg) {
        std::vector<Symbol> res(arg.size() + 1);
        Logarithm::create(arg.data(), res.data());
        return res;
    }

}
