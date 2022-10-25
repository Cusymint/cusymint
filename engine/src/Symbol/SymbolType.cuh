#ifndef SYMBOL_TYPE_CUH
#define SYMBOL_TYPE_CUH

namespace Sym {
    enum class Type : size_t {
        // Basic types
        Symbol = 0,
        Unknown,
        Variable,
        NumericConstant, // Constants with given numeric value, e.g. 5, 1.345, 12.44
        KnownConstant,   // Well known real constants, e.g. pi, e (Euler's number)
        UnknownConstant, // Constants marked with letters, e.g. a, phi, delta
        // Placeholders
        ExpanderPlaceholder,
        // Meta structures
        Integral,
        Solution,
        Substitution,
        SubexpressionVacancy,
        SubexpressionCandidate,
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
        // Logarithm
        Logarithm
    };

    __host__ __device__ inline const char* type_name(Type type) {
        // I could not find a better way to this that would work on both CPU and GPU
        switch (type) {
        case Type::Symbol:
            return "Symbol";
        case Type::Unknown:
            return "Unknown";
        case Type::Variable:
            return "Variable";
        case Type::NumericConstant:
            return "NumericConstant";
        case Type::KnownConstant:
            return "KnownConstant";
        case Type::UnknownConstant:
            return "UnknownConstant";
        case Type::ExpanderPlaceholder:
            return "ExpanderPlaceholder";
        case Type::Integral:
            return "Integral";
        case Type::Solution:
            return "Solution";
        case Type::Substitution:
            return "Substitution";
        case Type::SubexpressionVacancy:
            return "SubexpressionVacancy";
        case Type::SubexpressionCandidate:
            return "SubexpressionCandidate";
        case Type::Addition:
            return "Addition";
        case Type::Negation:
            return "Negation";
        case Type::Product:
            return "Product";
        case Type::Reciprocal:
            return "Reciprocal";
        case Type::Power:
            return "Power";
        case Type::Sine:
            return "Sine";
        case Type::Cosine:
            return "Cosine";
        case Type::Tangent:
            return "Tangent";
        case Type::Cotangent:
            return "Cotangent";
        case Type::Arcsine:
            return "Arcsine";
        case Type::Arccosine:
            return "Arccosine";
        case Type::Arctangent:
            return "Arctangent";
        case Type::Arccotangent:
            return "Arccotangent";
        case Type::Logarithm:
            return "Logarithm";
        }

        return "Invalid type";
    }
}

#endif
