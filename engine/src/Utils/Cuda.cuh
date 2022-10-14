#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cassert>
#include <cstddef>

namespace Util {
    constexpr double eps = 1e-10;

    __device__ inline size_t thread_count() { return gridDim.x * blockDim.x; }
    __device__ inline size_t thread_idx() { return threadIdx.x + blockDim.x * blockIdx.x; }

    /*
     * @brief GPU compatible alternative to memcmp
     *
     * @param p1 First memory pointer
     * @param p2 Second memory pointer
     * @param n Number of bytes to compare
     *
     * @return `true` if blocks `p1` and `p2` are equal, `false` otherwise
     */
    __host__ __device__ bool compare_mem(const void* const mem1, const void* const mem2,
                                         const size_t n);

    /*
     * @brief GPU compatible alternative to memcpy
     *
     * @param dst Destination of the copy
     * @param src Source of the copy
     * @param n Number of bytes to copy
     */
    __host__ __device__ void copy_mem(void* const dst, const void* const src, const size_t n);

    /*
     * @brief Swaps two memory blocks
     *
     * @param p1 First memory pointer
     * @param p2 Second memory pointer
     * @param n Number of bytes to swap
     */
    __host__ __device__ void swap_mem(void* const mem1, void* const mem2, const size_t n);

    /*
     * @brief Crashuje program w przypadku nieodwracalny błędów.
     *
     * @param message Wiadomość wypisywania na stdout.
     */
    __host__ __device__ void crash(const char* const message);
}

#endif
