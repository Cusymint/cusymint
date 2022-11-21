#ifndef SIMPLIFICATION_RESULT_H
#define SIMPLIFICATION_RESULT_H

namespace Sym {
    enum SimplificationResult { Success, Failure, NeedsSpace, NeedsSimplification };

    /*
     * @brief Retrieves information if `simplify` loop has to be run once more.
     *
     * @param `result` Result of a simplification
     *
     * @return `true` if `simplify` has to be run again, `false` otherwise.
     */
    __host__ __device__ inline bool is_another_loop_required(SimplificationResult result) {
        return result == SimplificationResult::NeedsSpace || result == SimplificationResult::NeedsSimplification;
    }
}

#endif