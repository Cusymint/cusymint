#include "Cuda.cuh"

#include <cstdint>

#include <iostream>
#include <memory>

namespace Util {
    // TODO: Sprawdzać alignment w funkcjach operujących na pamięci

    __host__ __device__ bool compare_mem(const void* const mem1, const void* const mem2,
                                         const size_t n) {
        // Porównanie bloków po 8 bajtów, potem porównanie pozostałych bajt po bajcie
        const auto* const p1_64 = reinterpret_cast<const uint64_t*>(mem1);
        const auto* const p2_64 = reinterpret_cast<const uint64_t*>(mem2);

        for (size_t i = 0; i < n / 8; ++i) {
            if (p1_64[i] != p2_64[i]) {
                return false;
            }
        }

        const auto* const p1_8 = reinterpret_cast<const uint8_t*>(mem1);
        const auto* const p2_8 = reinterpret_cast<const uint8_t*>(mem2);

        for (size_t i = n - n % 8; i < n; ++i) {
            if (p1_8[i] != p2_8[i]) {
                return false;
            }
        }

        return true;
    }

    __host__ __device__ void copy_mem(void* const dst, const void* const src, const size_t n) {
        // Kopiowanie bloków po 8 bajtów, potem kopiowanie pozostałych bajt po bajcie.
        auto* const dst_64 = reinterpret_cast<uint64_t*>(dst);
        const auto* const src_64 = reinterpret_cast<const uint64_t*>(src);

        for (size_t i = 0; i < n / 8; ++i) {
            dst_64[i] = src_64[i];
        }

        auto* const dst_8 = reinterpret_cast<uint8_t*>(dst);
        const auto* const src_8 = reinterpret_cast<const uint8_t*>(src);

        for (size_t i = n - n % 8; i < n; ++i) {
            dst_8[i] = src_8[i];
        }
    }

    __host__ __device__ void swap_mem(void* const mem1, void* const mem2, const size_t n) {
        // Swap bloków po 8 bajtów, potem swap pozostałych bajt po bajcie
        auto* const p1_64 = reinterpret_cast<uint64_t*>(mem1);
        auto* const p2_64 = reinterpret_cast<uint64_t*>(mem2);

        for (size_t i = 0; i < n / 8; ++i) {
            uint64_t temp = p1_64[i];
            p1_64[i] = p2_64[i];
            p2_64[i] = temp;
        }

        auto* const p1_8 = reinterpret_cast<uint8_t*>(mem1);
        auto* const p2_8 = reinterpret_cast<uint8_t*>(mem2);

        for (size_t i = n - n % 8; i < n; ++i) {
            uint8_t temp = p1_8[i];
            p1_8[i] = p2_8[i];
            p2_8[i] = temp;
        }
    }

    __host__ __device__ void crash(const char* const message) {
        printf("\n");
        printf("%s\n", message);
        assert(false);
    }
}
