#ifndef INTEGRATE_CUH
#define INTEGRATE_CUH

#include "symbol.cuh"

namespace Sym {
    typedef size_t (*ApplicabilityCheck)(Symbol* expression);
    typedef void (*ExpressionTransform)(Symbol* source, Symbol* destination);

    __device__ size_t is_simple_variable_power(Symbol* expression);
    __device__ size_t is_variable_exponent(Symbol* expression);
    __device__ size_t is_simple_sine(Symbol* expression);
    __device__ size_t is_simple_cosine(Symbol* expression);
    __device__ size_t is_sum(Symbol* expression);

    __device__ void integrate_simple_variable_power(Symbol* expression, Symbol* destination);
    __device__ void integrate_variable_exponent(Symbol* expression, Symbol* destination);
    __device__ void integrate_simple_sine(Symbol* expression, Symbol* destination);
    __device__ void integrate_simple_cosine(Symbol* expression, Symbol* destination);
    __device__ void integrate_sum(Symbol* expression, Symbol* destination);

    // HEURISTIC_CHECK_COUNT cannot be defined as sizeof(heurisic_checks) because `heurisic_checks`
    // is defined in translation unit associated with integrate.cu, but its size has to be known in
    // other translation units as well
    constexpr size_t KNOWN_INTEGRAL_COUNT = 4;
    constexpr size_t HEURISTIC_CHECK_COUNT = 1;
    constexpr size_t MAX_CHECK_COUNT =
        KNOWN_INTEGRAL_COUNT > HEURISTIC_CHECK_COUNT ? KNOWN_INTEGRAL_COUNT : HEURISTIC_CHECK_COUNT;
    constexpr size_t TRANSFORM_GROUP_SIZE = 32;
    constexpr size_t MAX_EXPRESSION_COUNT = 32;
    constexpr size_t EXPRESSION_MAX_SYMBOL_COUNT = 1024;
    constexpr size_t APPLICABILITY_ARRAY_SIZE = MAX_CHECK_COUNT * MAX_EXPRESSION_COUNT;
    constexpr size_t EXPRESSION_ARRAY_SIZE = MAX_EXPRESSION_COUNT * EXPRESSION_MAX_SYMBOL_COUNT;

    __global__ void check_for_known_integrals(Symbol* expressions, size_t* applicability,
                                              size_t* expression_count);
    __global__ void apply_known_integrals(Symbol* expressions, Symbol* destinations,
                                          size_t* applicability, size_t* expression_count);
    __global__ void check_heuristics_applicability(Symbol* expressions, size_t* applicability,
                                                   size_t* expression_count);
    __global__ void apply_heuristics(Symbol* expressions, Symbol* destinations,
                                     size_t* applicability, size_t* expression_count);
}

#endif
