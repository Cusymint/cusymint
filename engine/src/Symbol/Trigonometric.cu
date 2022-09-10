#include "Trigonometric.cuh"

#include "Symbol.cuh"

namespace Sym {
    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(Sine)
    DEFINE_SIMPLE_ONE_ARGUMETN_OP_COMPARE(Sine)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Sine)

    DEFINE_SIMPLIFY_IN_PLACE(Sine) { arg().simplify_in_place(help_space); }

    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(Cosine)
    DEFINE_SIMPLE_ONE_ARGUMETN_OP_COMPARE(Cosine)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Cosine)

    DEFINE_SIMPLIFY_IN_PLACE(Cosine) { arg().simplify_in_place(help_space); }

    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(Tangent)
    DEFINE_SIMPLE_ONE_ARGUMETN_OP_COMPARE(Tangent)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Tangent)

    DEFINE_SIMPLIFY_IN_PLACE(Tangent) { arg().simplify_in_place(help_space); }

    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(Cotangent)
    DEFINE_SIMPLE_ONE_ARGUMETN_OP_COMPARE(Cotangent)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Cotangent)

    DEFINE_SIMPLIFY_IN_PLACE(Cotangent) { arg().simplify_in_place(help_space); }

    std::string Sine::to_string() const { return "sin(" + arg().to_string() + ")"; }

    std::string Cosine::to_string() const { return "cos(" + arg().to_string() + ")"; }

    std::string Tangent::to_string() const { return "tan(" + arg().to_string() + ")"; }

    std::string Cotangent::to_string() const { return "cot(" + arg().to_string() + ")"; }

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
