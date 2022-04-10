#ifndef INTEGRATE_CUH
#define INTEGRATE_CUH

#include "symbol.cuh"

namespace Sym {
    constexpr size_t HEURISITC_GROUP_SIZE = 32;

    typedef bool (*HeuristicCheck)(Symbol* expression);

    __device__ bool is_simple_variable_power(Symbol* expression);
    __device__ bool is_variable_exponent(Symbol* expression);
    __device__ bool is_simple_sine(Symbol* expression);
    __device__ bool is_simple_cosine(Symbol* expression);
    __device__ bool is_sum(Symbol* expression);
    constexpr size_t HEURISTIC_CHECK_COUNT = 5;

    __global__ void check_heuristics_applicability(Symbol** expressions, bool** applicability,
                                                   size_t expression_count);
} // namespace Sym

#endif
