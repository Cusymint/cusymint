#include "Hyperbolic.h"

namespace Sym {
    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(SineHyperbolic)
    DEFINE_SIMPLE_ONE_ARGUMETN_OP_COMPARE(SineHyperbolic)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(SineHyperbolic)

    DEFINE_SIMPLIFY_IN_PLACE(SineHyperbolic) { arg().simplify_in_place(help_space); }

    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(CosineHyperbolic)
    DEFINE_SIMPLE_ONE_ARGUMETN_OP_COMPARE(CosineHyperbolic)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(CosineHyperbolic)

    DEFINE_SIMPLIFY_IN_PLACE(CosineHyperbolic) { arg().simplify_in_place(help_space); }

    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(TangentHyperbolic)
    DEFINE_SIMPLE_ONE_ARGUMETN_OP_COMPARE(TangentHyperbolic)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(TangentHyperbolic)

    DEFINE_SIMPLIFY_IN_PLACE(TangentHyperbolic) { arg().simplify_in_place(help_space); }

    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(CotangentHyperbolic)
    DEFINE_SIMPLE_ONE_ARGUMETN_OP_COMPARE(CotangentHyperbolic)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(CotangentHyperbolic)

    DEFINE_SIMPLIFY_IN_PLACE(CotangentHyperbolic) { arg().simplify_in_place(help_space); }

    std::string SineHyperbolic::to_string() const { return "sinh(" + arg().to_string() + ")"; }

    std::string CosineHyperbolic::to_string() const { return "cosh(" + arg().to_string() + ")"; }

    std::string TangentHyperbolic::to_string() const { return "tanh(" + arg().to_string() + ")"; }

    std::string CotangentHyperbolic::to_string() const { return "coth(" + arg().to_string() + ")"; }

    template <class T>
    std::vector<Symbol> make_hyperbolic_function(const T symbol,
                                                 const std::vector<Symbol>& arg) {
        std::vector<Symbol> result(arg.size() + 1);
        T::create(arg.data(), result.data());
        return result;
    }

    std::vector<Symbol> sinh(const std::vector<Symbol>& arg) {
        return make_hyperbolic_function(SineHyperbolic::create(), arg);
    }

    std::vector<Symbol> cosh(const std::vector<Symbol>& arg) {
        return make_hyperbolic_function(CosineHyperbolic::create(), arg);
    }

    std::vector<Symbol> tanh(const std::vector<Symbol>& arg) {
        return make_hyperbolic_function(TangentHyperbolic::create(), arg);
    }

    std::vector<Symbol> coth(const std::vector<Symbol>& arg) {
        return make_hyperbolic_function(CotangentHyperbolic::create(), arg);
    }
}