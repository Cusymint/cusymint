#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <limits>

namespace Util {
    /*
     * @brief The number of threads running in the current kernel
     */
    __device__ inline size_t thread_count() { return gridDim.x * blockDim.x; }

    /*
     * @brief A kernel-wide unique thread identifier
     */
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
     * @brief GPU compatible alternative to strncmp
     *
     * @param p1 First string pointer
     * @param p2 Second string pointer
     * @param n Maximal mumber of bytes to compare
     *
     * @return `true` if null-terminated strings `str1` and `str2` are equal
     * or their `n` first chars are equal, `false` otherwise
     */
    __host__ __device__ bool compare_str(const char* const str1, const char* const str2,
                                         const size_t n = std::numeric_limits<size_t>::max());


    /*
     * @brief Crashes the program in case of irreversible errors
     *
     * @param head Format string, same syntax as first argument of printf
     * @param tail Arguments used by the format string, same format as that of printf
     */

    template <class T, class... Types> __host__ __device__ void crash(T head, Types... tail) {
        printf("\n[ERROR]: ");
        printf(head, tail...);
        printf("\n");
        assert(false);
    }
}

#endif
