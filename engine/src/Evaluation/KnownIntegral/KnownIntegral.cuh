#ifndef KNOWN_INTEGRAL_CUH
#define KNOWN_INTEGRAL_CUH

#include "Evaluation/Status.cuh"

#include "Symbol/ExpressionArray.cuh"
#include "Symbol/MetaOperators.cuh"
#include "Symbol/Symbol.cuh"

namespace Sym::KnownIntegral {
    using Check = size_t (*)(const Integral& integral);
    using Application = EvaluationStatus (*)(const Integral& integral, SymbolIterator& destination,
                                             const ExpressionArray<>::Iterator& help_space);

    extern __device__ const Check CHECKS[];
    extern __device__ const Application APPLICATIONS[];

    /*
     * @brief Creates a solution of a desired type and checks if enough space is available
     *
     * @tparam MetaOperator Meta operator creating the solution (without the `SolutionOfIntegral`
     * operator
     *
     * @param destination Destination of the result
     * @param args Arguments passed to `SolutionOfIntegral<MetaOperator>`
     */
    template <class MetaOperator>
    __device__ EvaluationStatus
    simple_solution(SymbolIterator& destination,
                    const typename SolutionOfIntegral<MetaOperator>::AdditionalArgs& args) {
        using SolutionType = SolutionOfIntegral<MetaOperator>;

        ENSURE_ENOUGH_SPACE(SolutionType::size_with(args), destination);
        SolutionType::init(*destination, args);

        return EvaluationStatus::Done;
    }

#ifdef __CUDA_ARCH__
    __device__
#endif
        extern const size_t COUNT;
}

#endif
