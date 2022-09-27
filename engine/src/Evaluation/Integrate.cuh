#ifndef INTEGRATE_CUH
#define INTEGRATE_CUH

#include "Symbol/ExpressionArray.cuh"

#include "Symbol/Symbol.cuh"

namespace Sym {
    using ApplicabilityCheck = size_t (*)(const Integral* const);
    using IntegralTransform = void (*)(const Integral* const, Symbol* const, Symbol* const);

    __device__ size_t is_single_variable(const Integral* const integral);
    __device__ size_t is_simple_variable_power(const Integral* const integral);
    __device__ size_t is_variable_exponent(const Integral* const integral);
    __device__ size_t is_simple_sine(const Integral* const integral);
    __device__ size_t is_simple_cosine(const Integral* const integral);
    __device__ size_t is_constant(const Integral* const integral);
    __device__ size_t is_known_arctan(const Integral* const integral);

    __device__ void integrate_single_variable(const Integral* const integral,
                                              Symbol* const destination, Symbol* const help_space);
    __device__ void integrate_simple_variable_power(const Integral* const integral,
                                                    Symbol* const destination,
                                                    Symbol* const help_space);
    __device__ void integrate_variable_exponent(const Integral* const integral,
                                                Symbol* const destination,
                                                Symbol* const help_space);
    __device__ void integrate_simple_sine(const Integral* const integral, Symbol* const destination,
                                          Symbol* const help_space);
    __device__ void integrate_simple_cosine(const Integral* const integral,
                                            Symbol* const destination, Symbol* const help_space);
    __device__ void integrate_constant(const Integral* const integral, Symbol* const destination,
                                       Symbol* const help_space);
    __device__ void integrate_arctan(const Integral* const integral, Symbol* const destination,
                                     Symbol* const help_space);

    __device__ size_t is_function_of_ex(const Integral* const integral);

    __device__ void transform_function_of_ex(const Integral* const integral,
                                             Symbol* const destination, Symbol* const help_space);

    /*
     * @brief Tworzy symbol `Solution` i zapisuje go na `destination` razem z podstawieniami z
     * `integral`
     *
     * @param integral Całka z której skopiowane mają być podstawienia
     * @param destination Miejsce do zapisania wyniku
     *
     * @return Wskaźnik na symbol za ostatnim podstawieniem
     */
    __device__ Symbol* prepare_solution(const Integral* const integral, Symbol* const destination);

    // HEURISTIC_CHECK_COUNT cannot be defined as sizeof(heurisic_checks) because
    // `heurisic_checks` is defined in translation unit associated with integrate.cu, but its
    // size has to be known in other translation units as well
    constexpr size_t KNOWN_INTEGRAL_COUNT = 7;
    constexpr size_t HEURISTIC_CHECK_COUNT = 1;
    constexpr size_t MAX_CHECK_COUNT =
        KNOWN_INTEGRAL_COUNT > HEURISTIC_CHECK_COUNT ? KNOWN_INTEGRAL_COUNT : HEURISTIC_CHECK_COUNT;
    constexpr size_t TRANSFORM_GROUP_SIZE = 32;
    constexpr size_t MAX_INTEGRAL_COUNT = 256;
    constexpr size_t INTEGRAL_MAX_SYMBOL_COUNT = 256;
    constexpr size_t APPLICABILITY_ARRAY_SIZE = MAX_CHECK_COUNT * MAX_INTEGRAL_COUNT;
    constexpr size_t INTEGRAL_ARRAY_SIZE = MAX_INTEGRAL_COUNT * INTEGRAL_MAX_SYMBOL_COUNT;

    __global__ void check_for_known_integrals(const ExpressionArray<Integral> integrals,
                                              Util::DeviceArray<size_t> applicability);
    __global__ void apply_known_integrals(const ExpressionArray<Integral> integrals,
                                          ExpressionArray<> destinations,
                                          ExpressionArray<> help_spaces,
                                          const Util::DeviceArray<size_t> applicability);

    __global__ void check_heuristics_applicability(const ExpressionArray<Integral> integrals,
                                                   Util::DeviceArray<size_t> applicability);
    __global__ void apply_heuristics(const ExpressionArray<Integral> integrals,
                                     ExpressionArray<> destinations, ExpressionArray<> help_spaces,
                                     const Util::DeviceArray<size_t> applicability);

    __global__ void simplify(ExpressionArray<> expressions, ExpressionArray<> help_spaces);
}

#endif
