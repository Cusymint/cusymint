#include <iostream>
#include <vector>

namespace Sym {

    enum class Type {
        // Basic types
        Unknown,
        Variable,
        Constant,
        // Arithmetic
        Addition,
        Negative,
        Product,
        Reciprocal,
        Power,
        // Trigonometric functions
        Sine,
        Cosine,
        Tangent,
        Cotangent
    };

#define DECLARE_SYMBOL(_name) \
    struct _name {            \
        Sym::Type type;       \
                              \
        static _name create() { return {.type = Sym::Type::_name}; }

// A struct is POD iff it is standard-layout and trivial.
// standard-layout is required to guarantee that all symbolic types have the `type` member
// at offset 0 in the memory layout (necessary for later use in Symbol union).
// trivial is necessary for copying symbolic structs using simple memcpy calls
#define END_DECLARE_SYMBOL(_name) \
    }                             \
    ;                             \
    static_assert(std::is_pod<_name>::value, "Type _name has to be POD, but is not!");

#define ONE_ARGUMENT_OP_SYMBOL size_t total_size;

#define TWO_ARGUMENT_OP_SYMBOL \
    size_t total_size;         \
    size_t second_arg;

    DECLARE_SYMBOL(Unknown)
    END_DECLARE_SYMBOL(Unknown)

    DECLARE_SYMBOL(Variable)
    END_DECLARE_SYMBOL(Variable)

    DECLARE_SYMBOL(Constant)
    double value;
    END_DECLARE_SYMBOL(Constant)

    DECLARE_SYMBOL(Addition)
    TWO_ARGUMENT_OP_SYMBOL
    END_DECLARE_SYMBOL(Addition)

    DECLARE_SYMBOL(Negative)
    ONE_ARGUMENT_OP_SYMBOL
    END_DECLARE_SYMBOL(Negative)

    DECLARE_SYMBOL(Product)
    TWO_ARGUMENT_OP_SYMBOL
    END_DECLARE_SYMBOL(Product)

    DECLARE_SYMBOL(Reciprocal)
    TWO_ARGUMENT_OP_SYMBOL
    END_DECLARE_SYMBOL(Reciprocal)

    DECLARE_SYMBOL(Power)
    TWO_ARGUMENT_OP_SYMBOL
    END_DECLARE_SYMBOL(Power)

    DECLARE_SYMBOL(Sine)
    ONE_ARGUMENT_OP_SYMBOL
    END_DECLARE_SYMBOL(Sine)

    DECLARE_SYMBOL(Cosine)
    ONE_ARGUMENT_OP_SYMBOL
    END_DECLARE_SYMBOL(Cosine)

    DECLARE_SYMBOL(Tangent)
    ONE_ARGUMENT_OP_SYMBOL
    END_DECLARE_SYMBOL(Tangent)

    DECLARE_SYMBOL(Cotangent)
    ONE_ARGUMENT_OP_SYMBOL
    END_DECLARE_SYMBOL(Cotangent)

    union Symbol {
        Unknown unknown;
        Variable variable;
        Constant constant;
        Addition addition;
        Negative negative;
        Product product;
        Reciprocal reciprocal;
        Power power;
        Sine sine;
        Cosine cosine;
        Tangent tangent;
        Cotangent cotangent;

        __device__ inline Type type() { return unknown.type; }

        __device__ inline bool is(Type type) { return unknown.type == type; }
    };

    __device__ bool is_simple_variable_exponent(Symbol* symbol) {
        return symbol[0].is(Type::Power) && symbol[1].is(Type::Variable) &&
               symbol[2].is(Type::Constant) && symbol[2].constant.value != 0.0;
    }

    __device__ bool is_variable_reciprocal(Symbol* symbol) {
        return symbol[0].is(Type::Reciprocal) &&
               symbol[1].is(Type::Constant) && symbol[2].is(Type::Variable);
    }

    typedef std::vector<Symbol> Integral;

} // namespace Sym

int main() {
    return 0;
}
