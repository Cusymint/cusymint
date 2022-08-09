#include "InverseTrigonometric.cuh"

#include "Symbol.cuh"

namespace Sym {
    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(Arcsine)
    DEFINE_SIMPLE_ONE_ARGUMETN_OP_COMPARE(Arcsine)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Arcsine)

    DEFINE_SIMPLIFY_IN_PLACE(Arcsine) { arg().simplify_in_place(help_space); }

    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(Arccosine)
    DEFINE_SIMPLE_ONE_ARGUMETN_OP_COMPARE(Arccosine)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Arccosine)

    DEFINE_SIMPLIFY_IN_PLACE(Arccosine) { arg().simplify_in_place(help_space); }

    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(Arctangent)
    DEFINE_SIMPLE_ONE_ARGUMETN_OP_COMPARE(Arctangent)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Arctangent)

    DEFINE_SIMPLIFY_IN_PLACE(Arctangent) { arg().simplify_in_place(help_space); }

    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(Arccotangent)
    DEFINE_SIMPLE_ONE_ARGUMETN_OP_COMPARE(Arccotangent)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Arccotangent)

    DEFINE_SIMPLIFY_IN_PLACE(Arccotangent) { arg().simplify_in_place(help_space); }

    std::string Arcsine::to_string() const { return "arcsin(" + arg().to_string() + ")"; }

    std::string Arccosine::to_string() const { return "arccos(" + arg().to_string() + ")"; }

    std::string Arctangent::to_string() const { return "arctan(" + arg().to_string() + ")"; }

    std::string Arccotangent::to_string() const { return "arccot(" + arg().to_string() + ")"; }

    template <class T>
    std::vector<Symbol> make_inverse_trigonometric_function(const T symbol,
                                                            const std::vector<Symbol>& arg) {
        std::vector<Symbol> result(arg.size() + 1);
        T::create(arg.data(), result.data());
        return result;
    }

    std::vector<Symbol> arcsin(const std::vector<Symbol>& arg) {
        return make_inverse_trigonometric_function(Arcsine::create(), arg);
    }

    std::vector<Symbol> arccos(const std::vector<Symbol>& arg) {
        return make_inverse_trigonometric_function(Arccosine::create(), arg);
    }

    std::vector<Symbol> arctan(const std::vector<Symbol>& arg) {
        return make_inverse_trigonometric_function(Arctangent::create(), arg);
    }

    std::vector<Symbol> arccot(const std::vector<Symbol>& arg) {
        return make_inverse_trigonometric_function(Arccotangent::create(), arg);
    }
}
