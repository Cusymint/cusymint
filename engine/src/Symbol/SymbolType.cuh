#ifndef SYMBOL_TYPE_CUH
#define SYMBOL_TYPE_CUH

namespace Sym {
    enum class Type : size_t {
        // Basic types
        Unknown = 0,
        Variable,
        NumericConstant, // rational constants with given numeric value, e.g. 5, 1.345, 12.44
        KnownConstant,   // well known real constants, e.g. pi, e (Euler's number)
        UnknownConstant, // constants marked with letters, e.g. a, phi, delta
        // Placeholders
        ExpanderPlaceholder,
        // Meta structures
        Integral,
        Solution,
        Substitution,
        // Arithmetic
        Addition,
        Negation,
        Product,
        Reciprocal,
        Power,
        // Trigonometric functions
        Sine,
        Cosine,
        Tangent,
        Cotangent,
        // Inverse trigonometric functions
        Arcsine,
        Arccosine,
        Arctangent,
        Arccotangent,
        // Polynomial
        Polynomial
    };
}

#endif
