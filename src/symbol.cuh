#ifndef SYMBOL_CUH
#define SYMBOL_CUH

#include <string>
#include <vector>

#include "addition.cuh"
#include "constants.cuh"
#include "power.cuh"
#include "product.cuh"
#include "trigonometric.cuh"
#include "unknown.cuh"
#include "variable.cuh"

namespace Sym {
    union Symbol {
        Unknown unknown;
        Variable variable;
        NumericConstant numeric_constant;
        KnownConstant known_constant;
        UnknownConstant unknown_constant;
        Addition addition;
        Negative negative;
        Product product;
        Reciprocal reciprocal;
        Power power;
        Sine sine;
        Cosine cosine;
        Tangent tangent;
        Cotangent cotangent;

        __host__ __device__ inline Type type() const { return unknown.type; }
        __host__ __device__ inline bool is(Type type) const { return unknown.type == type; }

        template <class T> __host__ __device__ inline T& as() {
            return *reinterpret_cast<T*>(this);
        }

        template <class T> __host__ __device__ static inline Symbol* from(T* sym) {
            return reinterpret_cast<Symbol*>(sym);
        }

        __host__ __device__ bool is_simple_variable_polynomial() const;
        __host__ __device__ bool is_variable_reciprocal() const;
        __host__ __device__ bool is_numeric_constant_addition() const;
        __host__ __device__ bool is_numeric_constant_product() const;
        __host__ __device__ bool is_numeric_constant_negation() const;
        __host__ __device__ void simplify_in_place();

        __host__ std::string to_string();
    };

#define VIRTUAL_CALL(_instance, _member_function, ...)                         \
    (([&]() {                                                                  \
        switch ((_instance).unknown.type) {                                    \
        case Type::Variable:                                                   \
            return (_instance).variable._member_function(__VA_ARGS__);         \
        case Type::NumericConstant:                                            \
            return (_instance).numeric_constant._member_function(__VA_ARGS__); \
        case Type::KnownConstant:                                              \
            return (_instance).known_constant._member_function(__VA_ARGS__);   \
        case Type::UnknownConstant:                                            \
            return (_instance).unknown_constant._member_function(__VA_ARGS__); \
        case Type::Addition:                                                   \
            return (_instance).addition._member_function(__VA_ARGS__);         \
        case Type::Negative:                                                   \
            return (_instance).negative._member_function(__VA_ARGS__);         \
        case Type::Product:                                                    \
            return (_instance).product._member_function(__VA_ARGS__);          \
        case Type::Reciprocal:                                                 \
            return (_instance).reciprocal._member_function(__VA_ARGS__);       \
        case Type::Power:                                                      \
            return (_instance).power._member_function(__VA_ARGS__);            \
        case Type::Sine:                                                       \
            return (_instance).sine._member_function(__VA_ARGS__);             \
        case Type::Cosine:                                                     \
            return (_instance).cosine._member_function(__VA_ARGS__);           \
        case Type::Tangent:                                                    \
            return (_instance).tangent._member_function(__VA_ARGS__);          \
        case Type::Cotangent:                                                  \
            return (_instance).cotangent._member_function(__VA_ARGS__);        \
        case Type::Unknown:                                                    \
        default:                                                               \
            return (_instance).unknown._member_function(__VA_ARGS__);          \
        }                                                                      \
    })())
} // namespace Sym

#endif
