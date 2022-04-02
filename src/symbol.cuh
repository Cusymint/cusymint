#ifndef SYMBOL_CUH
#define SYMBOL_CUH

#include <string>
#include <vector>

#include "unknown.cuh"
#include "constants.cuh"
#include "variable.cuh"
#include "addition.cuh"
#include "product.cuh"
#include "power.cuh"
#include "trigonometric.cuh"

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

        __host__ __device__ inline Type type() const;
        __host__ __device__ inline bool is(Type type) const;
        template <class T> __host__ __device__ inline T& as() {
            return *reinterpret_cast<T*>(this);
        }

        __host__ __device__ bool is_simple_variable_polynomial() const;
        __host__ __device__ bool is_variable_reciprocal() const;
        __host__ __device__ bool is_numeric_constant_addition() const;
        __host__ __device__ bool is_numeric_constant_product() const;
        __host__ __device__ bool is_numeric_constant_negation() const;

        __host__ std::string to_string();
    };

} // namespace Sym

#endif
