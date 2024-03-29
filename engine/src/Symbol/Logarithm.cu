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

    DEFINE_SIMPLIFY_IN_PLACE(Logarithm) { // NOLINT(misc-unused-parameters)
        if (arg().is(Type::NumericConstant) && arg().as<NumericConstant>().value == 1) {
            // ln(1) = 0
            symbol().init_from(NumericConstant::with_value(0));
            return true;
        }

        if (arg().is(Type::KnownConstant) &&
            arg().as<KnownConstant>().value == KnownConstantValue::E) {
            // ln(e) = 1
            symbol().init_from(NumericConstant::with_value(1));
            return true;
        }

        if (arg().is(Type::Power)) {
            // Here we do the following transformation: ln(f(x)^g(x)) = ln(f(x))*g(x).
            // Structure in memory before operation: LN | ^  | f(x) | g(x)
            // Structure after operation:            *  | LN | f(x) | g(x)
            // Therefore, it is not necessary to do any copying, we only need to
            // change types of 2 first symbols, size of the second and add second_arg_offset
            // to new Product symbol.
            const auto offset = arg().power.second_arg_offset;
            arg().power.type = Type::Logarithm;
            arg().as<Logarithm>().seal();
            type = Type::Product;
            symbol().as<Product>().second_arg_offset =
                offset + 1; // seal_arg_1 may not work in this case (e.g. size of f(x) changes)
            return false;   // resulting Product is not sorted
        }

        if (arg().is(Type::Product)) {
            const auto count = arg().as<Product>().tree_size();
            if (size < arg().size() + count) {
                additional_required_size = count - 1;
                return false;
            }
            From<Product>::Create<Addition>::WithMap<Ln>::init(*help_space,
                                                               {{arg().as<Product>(), count}});
            help_space->copy_to(symbol());
            return false;
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
