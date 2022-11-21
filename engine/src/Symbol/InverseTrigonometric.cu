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
        return Mul<Inv<Pow<Add<Integer<1>, Neg<Pow<Copy, Integer<2>>>>, Num>>, None>::init_reverse(*destination, {arg(), 0.5});
    }

    DEFINE_INSERT_REVERSED_DERIVATIVE_AT(Arccosine) {
        if ((destination - 1)->is(0)) {
            return 0;
        }
        return Mul<Neg<Inv<Pow<Add<Integer<1>, Neg<Pow<Copy, Integer<2>>>>, Num>>>, None>::init_reverse(*destination, {arg(), 0.5});
    }

    DEFINE_INSERT_REVERSED_DERIVATIVE_AT(Arctangent) {
        if ((destination - 1)->is(0)) {
            return 0;
        }
        return Mul<Inv<Add<Integer<1>, Pow<Copy, Integer<2>>>>, None>::init_reverse(*destination, {arg()});
    }

    DEFINE_INSERT_REVERSED_DERIVATIVE_AT(Arccotangent) {
        if ((destination - 1)->is(0)) {
            return 0;
        }
        return Mul<Neg<Inv<Add<Integer<1>, Pow<Copy, Integer<2>>>>>, None>::init_reverse(*destination, {arg()});
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
