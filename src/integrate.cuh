#ifndef INTEGRATE_CUH
#define INTEGRATE_CUH

#include "symbol.cuh"

namespace Sym {
    typedef size_t (*HeuristicCheck)(Symbol* expression);

    __device__ size_t is_simple_variable_power(Symbol* expression);
    __device__ size_t is_variable_exponent(Symbol* expression);
    __device__ size_t is_simple_sine(Symbol* expression);
    __device__ size_t is_simple_cosine(Symbol* expression);
    __device__ size_t is_sum(Symbol* expression);

    constexpr size_t HEURISTIC_CHECK_COUNT = 5;
    constexpr size_t HEURISITC_GROUP_SIZE = 32;
    constexpr size_t MAX_EXPRESSION_COUNT = 32;
    constexpr size_t EXPRESSION_MAX_SYMBOL_COUNT = 1024;
    constexpr size_t APPLICABILITY_ARRAY_SIZE = HEURISTIC_CHECK_COUNT * MAX_EXPRESSION_COUNT;
    constexpr size_t EXPRESSION_ARRAY_SIZE = MAX_EXPRESSION_COUNT * EXPRESSION_MAX_SYMBOL_COUNT;

    __global__ void check_heuristics_applicability(Symbol* expressions, size_t* applicability,
                                                   size_t expression_count);
} // namespace Sym

#endif
