#include "Cuda.cuh"

#include <cstdint>

namespace Util {
    __host__ __device__ bool compare_mem(const void* const mem1, const void* const mem2,
                                         const size_t n) {
        const auto* const p1_8 = reinterpret_cast<const uint8_t*>(mem1);
        const auto* const p2_8 = reinterpret_cast<const uint8_t*>(mem2);

        for (size_t i = 0; i < n; ++i) {
            if (p1_8[i] != p2_8[i]) {
                return false;
            }
        }

        return true;
    }

    __host__ __device__ void copy_mem(void* const dst, const void* const src, const size_t n) {
        auto* const dst_8 = reinterpret_cast<uint8_t*>(dst);
        const auto* const src_8 = reinterpret_cast<const uint8_t*>(src);

        for (size_t i = 0; i < n; ++i) {
            dst_8[i] = src_8[i];
        }
    }

    __host__ __device__ void swap_mem(void* const mem1, void* const mem2, const size_t n) {
        auto* const p1_8 = reinterpret_cast<uint8_t*>(mem1);
        auto* const p2_8 = reinterpret_cast<uint8_t*>(mem2);

        for (size_t i = 0; i < n; ++i) {
            uint8_t temp = p1_8[i];
            p1_8[i] = p2_8[i];
            p2_8[i] = temp;
        }
    }
}
