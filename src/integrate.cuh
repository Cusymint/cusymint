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
    static constexpr size_t MAX_EXPRESSION_COUNT = 32;
    static constexpr size_t APPLICABILITY_SIZE = HEURISTIC_CHECK_COUNT * MAX_EXPRESSION_COUNT;

    __global__ void check_heuristics_applicability(Symbol** expressions, size_t* applicability,
                                                   size_t expression_count);
} // namespace Sym

#endif
