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
        Arccotangent
    };

    __host__ __device__ inline const char* type_name(Type type) {
        // I could not find a better way to this that would work on both CPU and GPU
        switch (type) {
        case Type::Symbol:
            return "Symbol";
            break;
        case Type::Unknown:
            return "Unknown";
            break;
        case Type::Variable:
            return "Variable";
            break;
        case Type::NumericConstant:
            return "NumericConstant";
            break;
        case Type::KnownConstant:
            return "KnownConstant";
            break;
        case Type::UnknownConstant:
            return "UnknownConstant";
            break;
        case Type::ExpanderPlaceholder:
            return "ExpanderPlaceholder";
            break;
        case Type::Integral:
            return "Integral";
            break;
        case Type::Solution:
            return "Solution";
            break;
        case Type::Substitution:
            return "Substitution";
            break;
        case Type::SubexpressionVacancy:
            return "SubexpressionVacancy";
            break;
        case Type::SubexpressionCandidate:
            return "SubexpressionCandidate";
            break;
        case Type::Addition:
            return "Addition";
            break;
        case Type::Negation:
            return "Negation";
            break;
        case Type::Product:
            return "Product";
            break;
        case Type::Reciprocal:
            return "Reciprocal";
            break;
        case Type::Power:
            return "Power";
            break;
        case Type::Sine:
            return "Sine";
            break;
        case Type::Cosine:
            return "Cosine";
            break;
        case Type::Tangent:
            return "Tangent";
            break;
        case Type::Cotangent:
            return "Cotangent";
            break;
        case Type::Arcsine:
            return "Arcsine";
            break;
        case Type::Arccosine:
            return "Arccosine";
            break;
        case Type::Arctangent:
            return "Arctangent";
            break;
        case Type::Arccotangent:
            return "Arccotangent";
            break;
        }

        return "Invalid type";
    }
}

#endif
