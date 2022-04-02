#ifndef SYMBOL_DEFS_CUH
#define SYMBOL_DEFS_CUH

#include <string>
#include <type_traits>

namespace Sym {
    union Symbol;

    enum class Type {
        // Basic types
        Unknown,
        Variable,
        NumericConstant, // rational constants with given numeric value, e.g. 5, 1.345, 12.44
        KnownConstant,   // well known real constants, e.g. pi, e (Euler's number)
        UnknownConstant, // constants marked with letters, e.g. a, phi, delta
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

#define DECLARE_SYMBOL(_name, _simple)    \
    struct _name {                        \
        Sym::Type type;                   \
        bool simplified;                  \
                                          \
        static _name create() {           \
            return {                      \
                .type = Sym::Type::_name, \
                .simplified = _simple,    \
            };                            \
        }

// A struct is POD iff it is standard-layout and trivial.
// standard-layout is required to guarantee that all symbolic types have the `type` member
// at offset 0 in the memory layout (necessary for later use in Symbol union).
// trivial is necessary for copying symbolic structs
#define END_DECLARE_SYMBOL(_name) \
    }                             \
    ;                             \
    static_assert(std::is_pod<_name>::value, "Type '" #_name "' has to be POD, but is not!");

#define ONE_ARGUMENT_OP_SYMBOL size_t total_size;

#define TWO_ARGUMENT_OP_SYMBOL \
    size_t total_size;         \
    size_t second_arg_offset;

#define DEFINE_TO_STRING(_str) \
    std::string to_string() { return _str; }

}; // namespace Sym

#endif
