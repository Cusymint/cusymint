#ifndef SYMBOL_TYPE_CUH
#define SYMBOL_TYPE_CUH

namespace Sym {
    enum class Type : size_t {
        // Basic types
        Symbol = 0,
        Unknown,
        // NumericConstant MUST be first here, some things depend on the fact that it is first in
        // the symbol ordering
        NumericConstant, // Constants with given numeric value, e.g. 5, 1.345, 12.44
        KnownConstant,   // Well known real constants, e.g. pi, e (Euler's number)
        UnknownConstant, // Constants marked with letters, e.g. a, phi, delta
        Variable,
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
        Product,
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
        Logarithm,
        // Polynomials
        Polynomial,
        // Error function
        ErrorFunction,
        // Integral functions
        SineIntegral,
        CosineIntegral,
        ExponentialIntegral,
        LogarithmicIntegral
    };

    /*
     * @brief A unique number for every symbol type that can be used for ordering
     *
     * @param type Type to get the number for
     *
     * @return A number uniquely identifying the type
     */
    __host__ __device__ inline size_t type_ordinal(const Type type) {
        return static_cast<size_t>(type);
    }

    __host__ __device__ inline const char* type_name(Type type) {
        // I could not find a better way to this that would work on both CPU and GPU
        switch (type) {
        case Type::Symbol:
            return "Symbol";
        case Type::Unknown:
            return "Unknown";
        case Type::NumericConstant:
            return "NumericConstant";
        case Type::KnownConstant:
            return "KnownConstant";
        case Type::UnknownConstant:
            return "UnknownConstant";
        case Type::Variable:
            return "Variable";
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
        case Type::Product:
            return "Product";
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
        case Type::Polynomial:
            return "Polynomial";
        case Type::ErrorFunction:
            return "ErrorFunction";
        case Type::SineIntegral:
            return "SineIntegral";
        case Type::CosineIntegral:
            return "CosineIntegral";
        case Type::ExponentialIntegral:
            return "ExponentialIntegral";
        case Type::LogarithmicIntegral:
            return "LogarithmicIntegral";
        }

        return "Invalid type";
    }
}

#endif
