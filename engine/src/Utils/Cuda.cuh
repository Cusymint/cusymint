#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace Util {
    /*
     * @brief Calculates the block count for a kernel given the mininum thread count and the block
     * size
     *
     * @param thread_count Minimum number of threads required
     * @param block_size Number of threads in each thread block
     */
    inline size_t block_count(const size_t thread_count, const size_t block_size) {
        return (thread_count - 1) / block_size + 1;
    }

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
     * @brief GPU compatible alternative to memcpy. `dst` and `src` cannot alias.
     *
     * @param dst Destination of the copy
     * @param src Source of the copy
     * @param n Number of bytes to copy
     */
    __host__ __device__ void copy_mem(void* const dst, const void* const src, const size_t n);

    /*
     * @brief Works like `copy_mem`, but its arguments can overlap and the original may be changed
     *
     * @param dst Destination of the copy
     * @param src Source of the copy
     * @param n Number of bytes to move
     */
    __host__ __device__ void move_mem(void* const dst, void* const src, const size_t n);

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
     * @brief Compares two values and returns the smaller one
     *
     * @param first First value to compare
     * @param second First value to compare
     *
     * @return the smaller of the two values
     */
    template <class T> __host__ __device__ T min(const T& first, const T& second) {
        return first < second ? first : second;
    }

    /*
     * @brief Compares two values and returns the greater one
     *
     * @param first First value to compare
     * @param second First value to compare
     *
     * @return the greater of the two values
     */
    template <class T> __host__ __device__ T max(const T& first, const T& second) {
        return first > second ? first : second;
    }

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
#ifdef __CUDA_ARCH__
        assert(false);
#else
        throw std::runtime_error("Fatal error");
#endif
    }
}

#endif
