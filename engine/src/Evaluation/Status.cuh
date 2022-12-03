#ifndef EVALUTAION_STATUS_CUH
#define EVALUTAION_STATUS_CUH

#include "Utils/Result.cuh"

/*
 * @brief Checks if enough memory is available for an operation. If yes, does nothing. If not,
 * returns `EvaluationStatus::ReallocationRequest` from the current function.
 *
 * @param _required Required amount of memory
 * @param _destination SymbolIterator or ExpressionIterator in which the memory has to be available
 */
#define ENSURE_ENOUGH_SPACE(_required, _destination)  \
    if ((_required) > (_destination).capacity()) {    \
        return EvaluationStatus::ReallocationRequest; \
    }

/*
 * @brief Checks if the result of `_call` is equal to `EvaluationStatus::Done`. If yes, does
 * nothing. If not, returns the result from the current function.
 */
#define TRY_EVALUATE(_call)                      \
    ({                                           \
        const EvaluationStatus result = (_call); \
        if (result != EvaluationStatus::Done) {  \
            return result;                       \
        }                                        \
                                                 \
        _call;                                   \
    })

/*
 * @brief Checks if the result of `_call` is a good result. If yes, does
 * nothing. If not, returns `EvaluationStatus::ReallocationRequest`.
 */
#define TRY_EVALUATE_RESULT(_call)                                          \
    ({                                                                      \
        const EvaluationStatus result = result_to_evaluation_status(_call); \
        if (result != EvaluationStatus::Done) {                             \
            return result;                                                  \
        }                                                                   \
                                                                            \
        (_call).good();                                                     \
    })

namespace Sym {
    enum class EvaluationStatus {
        Incomplete,
        Done,
        ReallocationRequest,
    };

    /*
     * @brief Converts a `Result` to `EvaluationStatus::Done` if the result is good, and to
     * `ReallocationRequest` if it is an error
     */
    template <class T, class E>
    [[nodiscard]] __host__ __device__ EvaluationStatus
    result_to_evaluation_status(const Util::Result<T, E>& result) {
        return result.is_good() ? EvaluationStatus::Done : EvaluationStatus::ReallocationRequest;
    }
}

#endif
