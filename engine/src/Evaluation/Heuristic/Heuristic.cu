#include "Heuristic.cuh"

#include "BringOutConstantsFromProduct.cuh"
#include "SplitSum.cuh"
#include "SubstituteEToX.cuh"
#include "UniversalSubstitution.cuh"

#include "Utils/Meta.cuh"

namespace Sym::Heuristic {
    // Try removing this function to convince yourself of the message contained in its name
    CheckResult NVLINK_IS_A_COMPLETE_JOKE() { return {0, 0}; }

    __device__ const Check CHECKS[] = {
        is_function_of_ex,
        is_sum,
        is_function_of_trigs,
        contains_constants_product,
    };

    __device__ const Application APPLICATIONS[] = {
        transform_function_of_ex,
        split_sum,
        do_universal_substitution,
        bring_out_constants_from_product,
    };

#ifdef __CUDA_ARCH__
    __device__
#endif
        const size_t COUNT =
            Util::ensure_same_v<Util::array_len(CHECKS), Util::array_len(APPLICATIONS)>;
}
