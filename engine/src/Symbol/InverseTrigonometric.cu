#include "InverseTrigonometric.cuh"

#include <fmt/core.h>

#include "Symbol.cuh"
#include "Symbol/Macros.cuh"
#include "Symbol/MetaOperators.cuh"
#include "Symbol/Power.cuh"
#include "Symbol/Product.cuh"
#include "Utils/Cuda.cuh"

namespace Sym {
    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(Arcsine)
    DEFINE_SIMPLE_ONE_ARGUMENT_OP_ARE_EQUAL(Arcsine)
    DEFINE_IDENTICAL_COMPARE_TO(Arcsine)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Arcsine)
    DEFINE_SIMPLE_ONE_ARGUMENT_IS_FUNCTION_OF(Arcsine)
    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(Arcsine)

    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(Arccosine)
    DEFINE_SIMPLE_ONE_ARGUMENT_OP_ARE_EQUAL(Arccosine)
    DEFINE_IDENTICAL_COMPARE_TO(Arccosine)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Arccosine)
    DEFINE_SIMPLE_ONE_ARGUMENT_IS_FUNCTION_OF(Arccosine)
    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(Arccosine)

    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(Arctangent)
    DEFINE_SIMPLE_ONE_ARGUMENT_OP_ARE_EQUAL(Arctangent)
    DEFINE_IDENTICAL_COMPARE_TO(Arctangent)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Arctangent)
    DEFINE_SIMPLE_ONE_ARGUMENT_IS_FUNCTION_OF(Arctangent)
    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(Arctangent)

    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(Arccotangent)
    DEFINE_SIMPLE_ONE_ARGUMENT_OP_ARE_EQUAL(Arccotangent)
    DEFINE_IDENTICAL_COMPARE_TO(Arccotangent)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Arccotangent)
    DEFINE_SIMPLE_ONE_ARGUMENT_IS_FUNCTION_OF(Arccotangent)
    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(Arccotangent)

    DEFINE_INSERT_REVERSED_DERIVATIVE_AT(Arcsine) {
        if ((destination - 1)->is(0)) {
            return 0;
        }
        // (expr') 0.5 2 (expr) ^ - 1 + ^ inv *
        (destination)->init_from(NumericConstant::with_value(0.5));   // power exponent for sqrt
        (destination + 1)->init_from(NumericConstant::with_value(2)); // power exponent for x^2
        Symbol::copy_and_reverse_symbol_sequence(destination + 2, &arg(), arg().size());
        ManySymbols<Power, Negation, NumericConstant, Addition, Power, Reciprocal,
                    Product>::create_reversed_at(destination + arg().size() + 2);
        (destination + arg().size() + 4)->as<NumericConstant>().value = 1;
        return arg().size() + 9;
    }

    DEFINE_INSERT_REVERSED_DERIVATIVE_AT(Arccosine) {
        if ((destination - 1)->is(0)) {
            return 0;
        }
        // (expr') 0.5 2 (expr) ^ - 1 + ^ inv - *
        (destination)->init_from(NumericConstant::with_value(0.5));   // power exponent for sqrt
        (destination + 1)->init_from(NumericConstant::with_value(2)); // power exponent for x^2
        Symbol::copy_and_reverse_symbol_sequence(destination + 2, &arg(), arg().size());
        ManySymbols<Power, Negation, NumericConstant, Addition, Power, Reciprocal, Negation,
                    Product>::create_reversed_at(destination + arg().size() + 2);
        (destination + arg().size() + 4)->as<NumericConstant>().value = 1;
        return arg().size() + 10;
    }

    DEFINE_INSERT_REVERSED_DERIVATIVE_AT(Arctangent) {
        if ((destination - 1)->is(0)) {
            return 0;
        }
        // (expr') 2 (expr) ^ 1 + inv *
        (destination)->init_from(NumericConstant::with_value(2)); // power exponent for x^2
        Symbol::copy_and_reverse_symbol_sequence(destination + 1, &arg(), arg().size());
        Power::create_reversed_at(destination + arg().size() + 1);
        (destination + arg().size() + 2)->init_from(NumericConstant::with_value(1));
        ManySymbols<Addition, Reciprocal, Product>::create_reversed_at(destination + arg().size() +
                                                                       3);
        return arg().size() + 6;
    }

    DEFINE_INSERT_REVERSED_DERIVATIVE_AT(Arccotangent) {
        if ((destination - 1)->is(0)) {
            return 0;
        }
        // (expr') 2 (expr) ^ 1 + inv - *
        (destination)->init_from(NumericConstant::with_value(2)); // power exponent for x^2
        Symbol::copy_and_reverse_symbol_sequence(destination + 1, &arg(), arg().size());
        Power::create_reversed_at(destination + arg().size() + 1);
        (destination + arg().size() + 2)->init_from(NumericConstant::with_value(1));
        ManySymbols<Addition, Reciprocal, Negation, Product>::create_reversed_at(destination +
                                                                                 arg().size() + 3);
        return arg().size() + 7;
    }

    std::string Arcsine::to_string() const { return fmt::format("arcsin({})", arg().to_string()); }

    std::string Arccosine::to_string() const {
        return fmt::format("arccos({})", arg().to_string());
    }

    std::string Arctangent::to_string() const {
        return fmt::format("arctan({})", arg().to_string());
    }

    std::string Arccotangent::to_string() const {
        return fmt::format("arccot({})", arg().to_string());
    }

    std::string Arcsine::to_tex() const {
        return fmt::format(R"(\arcsin\left({}\right))", arg().to_tex());
    }

    std::string Arccosine::to_tex() const {
        return fmt::format(R"(\arccos\left({}\right))", arg().to_tex());
    }

    std::string Arctangent::to_tex() const {
        return fmt::format(R"(\arctan\left({}\right))", arg().to_tex());
    }

    std::string Arccotangent::to_tex() const {
        return fmt::format(R"(\arccot\left({}\right))", arg().to_tex());
    }

    template <class T>
    std::vector<Symbol> make_inverse_trigonometric_function(const std::vector<Symbol>& arg) {
        std::vector<Symbol> result(arg.size() + 1);
        T::create(arg.data(), result.data());
        return result;
    }

    std::vector<Symbol> arcsin(const std::vector<Symbol>& arg) {
        return make_inverse_trigonometric_function<Arcsine>(arg);
    }

    std::vector<Symbol> arccos(const std::vector<Symbol>& arg) {
        return make_inverse_trigonometric_function<Arccosine>(arg);
    }

    std::vector<Symbol> arctan(const std::vector<Symbol>& arg) {
        return make_inverse_trigonometric_function<Arctangent>(arg);
    }

    std::vector<Symbol> arccot(const std::vector<Symbol>& arg) {
        return make_inverse_trigonometric_function<Arccotangent>(arg);
    }
}
