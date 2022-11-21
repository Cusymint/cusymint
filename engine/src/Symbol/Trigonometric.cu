#include "Symbol/Macros.cuh"
#include "Symbol/Product.cuh"
#include "Trigonometric.cuh"

#include <fmt/core.h>

#include "Symbol.cuh"
#include "Symbol/MetaOperators.cuh"

namespace Sym {
    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(Sine)
    DEFINE_SIMPLE_ONE_ARGUMENT_OP_ARE_EQUAL(Sine)
    DEFINE_IDENTICAL_COMPARE_TO(Sine)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Sine)
    DEFINE_SIMPLE_ONE_ARGUMENT_IS_FUNCTION_OF(Sine)
    DEFINE_ONE_ARG_OP_DERIVATIVE(Sine, Cos<Copy>)

    DEFINE_SIMPLIFY_IN_PLACE(Sine) {
        if (arg().is(Type::Arcsine)) {
            arg().as<Arcsine>().arg().copy_to(help_space);
            help_space->copy_to(symbol());
        }
        return true;
    }

    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(Cosine)
    DEFINE_SIMPLE_ONE_ARGUMENT_OP_ARE_EQUAL(Cosine)
    DEFINE_IDENTICAL_COMPARE_TO(Cosine)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Cosine)
    DEFINE_SIMPLE_ONE_ARGUMENT_IS_FUNCTION_OF(Cosine)
    DEFINE_ONE_ARG_OP_DERIVATIVE(Cosine, Neg<Sin<Copy>>)

    DEFINE_SIMPLIFY_IN_PLACE(Cosine) {
        if (arg().is(Type::Arccosine)) {
            arg().as<Arccosine>().arg().copy_to(help_space);
            help_space->copy_to(symbol());
        }
        return true;
    }

    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(Tangent)
    DEFINE_SIMPLE_ONE_ARGUMENT_OP_ARE_EQUAL(Tangent)
    DEFINE_IDENTICAL_COMPARE_TO(Tangent)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Tangent)
    DEFINE_SIMPLE_ONE_ARGUMENT_IS_FUNCTION_OF(Tangent)
    DEFINE_ONE_ARG_OP_DERIVATIVE(Tangent, (Inv<Pow<Cos<Copy>, Integer<2>>>))

    DEFINE_SIMPLIFY_IN_PLACE(Tangent) {
        if (arg().is(Type::Arctangent)) {
            arg().as<Arctangent>().arg().copy_to(help_space);
            help_space->copy_to(symbol());
        }
        return true;
    }

    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(Cotangent)
    DEFINE_SIMPLE_ONE_ARGUMENT_OP_ARE_EQUAL(Cotangent)
    DEFINE_IDENTICAL_COMPARE_TO(Cotangent)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Cotangent)
    DEFINE_SIMPLE_ONE_ARGUMENT_IS_FUNCTION_OF(Cotangent)
    DEFINE_ONE_ARG_OP_DERIVATIVE(Cotangent, (Neg<Inv<Pow<Sin<Copy>, Integer<2>>>>))

    DEFINE_SIMPLIFY_IN_PLACE(Cotangent) {
        if (arg().is(Type::Arccotangent)) {
            arg().as<Arccotangent>().arg().copy_to(help_space);
            help_space->copy_to(symbol());
        }
        return true;
    }

    std::string Sine::to_string() const { return fmt::format("sin({})", arg().to_string()); }

    std::string Cosine::to_string() const { return fmt::format("cos({})", arg().to_string()); }

    std::string Tangent::to_string() const { return fmt::format("tan({})", arg().to_string()); }

    std::string Cotangent::to_string() const { return fmt::format("cot({})", arg().to_string()); }

    std::string Sine::to_tex() const {
        return fmt::format(R"(\sin\left({}\right))", arg().to_tex());
    }

    std::string Cosine::to_tex() const {
        return fmt::format(R"(\cos\left({}\right))", arg().to_tex());
    }

    std::string Tangent::to_tex() const {
        return fmt::format(R"(\tan\left({}\right))", arg().to_tex());
    }

    std::string Cotangent::to_tex() const {
        return fmt::format(R"(\cot\left({}\right))", arg().to_tex());
    }

    template <class T> std::vector<Symbol> make_trig_function(const std::vector<Symbol>& arg) {
        std::vector<Symbol> res(arg.size() + 1);
        T::create(arg.data(), res.data());
        return res;
    }

    std::vector<Symbol> sin(const std::vector<Symbol>& arg) {
        return make_trig_function<Sine>(arg);
    }

    std::vector<Symbol> cos(const std::vector<Symbol>& arg) {
        return make_trig_function<Cosine>(arg);
    }

    std::vector<Symbol> tan(const std::vector<Symbol>& arg) {
        return make_trig_function<Tangent>(arg);
    }

    std::vector<Symbol> cot(const std::vector<Symbol>& arg) {
        return make_trig_function<Cotangent>(arg);
    }
}
