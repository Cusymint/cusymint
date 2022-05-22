#ifndef INTEGRATE_CUH
#define INTEGRATE_CUH

#include "symbol.cuh"

namespace Sym {
    typedef size_t (*ApplicabilityCheck)(const Integral* const integral);
    typedef void (*IntegralTransform)(const Integral* const integral, Symbol* const destination,
                                      Symbol* const swap_space);

    __device__ size_t is_simple_variable_power(const Integral* const integral);
    __device__ size_t is_variable_exponent(const Integral* const integral);
    __device__ size_t is_simple_sine(const Integral* const integral);
    __device__ size_t is_simple_cosine(const Integral* const integral);
    __device__ size_t is_constant(const Integral* const integral);

    __device__ void integrate_simple_variable_power(const Integral* const integral,
                                                    Symbol* const destination,
                                                    Symbol* const swap_space);
    __device__ void integrate_variable_exponent(const Integral* const integral,
                                                Symbol* const destination,
                                                Symbol* const swap_space);
    __device__ void integrate_simple_sine(const Integral* const integral, Symbol* const destination,
                                          Symbol* const swap_space);
    __device__ void integrate_simple_cosine(const Integral* const integral, Symbol* const destination,
                                            Symbol* const swap_space);
    __device__ void integrate_constant(const Integral* const integral, Symbol* const destination,
                                       Symbol* const swap_space);

    __device__ size_t is_function_of_ex(const Integral* const integral);

    __device__ void transform_function_of_ex(const Integral* const integral,
                                             Symbol* const destination, Symbol* const swap_space);

    /*
     * @brief Creates a `Solution` symbol at `destination[0]` and all substitutions from `integral`
     *
     * @param integral Integral to copy substitutions from
     * @param destination Solution destination
     *
     * @return Pointer to one symbol after last substitution (destination of solution symbols)
     */
    __device__ Symbol* prepare_solution(const Integral* const integral, Symbol* const destination);

    // HEURISTIC_CHECK_COUNT cannot be defined as sizeof(heurisic_checks) because
    // `heurisic_checks` is defined in translation unit associated with integrate.cu, but its
    // size has to be known in other translation units as well
    constexpr size_t KNOWN_INTEGRAL_COUNT = 5;
    constexpr size_t HEURISTIC_CHECK_COUNT = 1;
    constexpr size_t MAX_CHECK_COUNT =
        KNOWN_INTEGRAL_COUNT > HEURISTIC_CHECK_COUNT ? KNOWN_INTEGRAL_COUNT : HEURISTIC_CHECK_COUNT;
    constexpr size_t TRANSFORM_GROUP_SIZE = 32;
    constexpr size_t MAX_INTEGRAL_COUNT = 32;
    constexpr size_t INTEGRAL_MAX_SYMBOL_COUNT = 1024;
    constexpr size_t APPLICABILITY_ARRAY_SIZE = MAX_CHECK_COUNT * MAX_INTEGRAL_COUNT;
    constexpr size_t INTEGRAL_ARRAY_SIZE = MAX_INTEGRAL_COUNT * INTEGRAL_MAX_SYMBOL_COUNT;

    __global__ void check_for_known_integrals(const Symbol* const integrals,
                                              size_t* const applicability,
                                              const size_t* const integral_count);
    __global__ void apply_known_integrals(const Symbol* const integrals, Symbol* const destinations,
                                          Symbol* const swap_spaces,
                                          const size_t* const applicability,
                                          const size_t* const integral_count);

    __global__ void check_heuristics_applicability(const Symbol* const integrals,
                                                   size_t* const applicability,
                                                   const size_t* const integral_count);
    __global__ void apply_heuristics(const Symbol* const integrals, Symbol* const destinations,
                                     Symbol* const swap_spaces, const size_t* const applicability,
                                     const size_t* const integral_count);
}

#endif
