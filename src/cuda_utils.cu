#include "cuda_utils.cuh"

#include <cstdint>

#include <iostream>
#include <memory>

namespace Util {
    // TODO: Sprawdzać alignment w funkcjach operujących na pamięci

    __host__ __device__ bool compare_mem(const void* const p1, const void* const p2,
                                         const size_t n) {
        // Porównanie bloków po 8 bajtów, potem porównanie pozostałych bajt po bajcie
        const uint64_t* const p1_64 = reinterpret_cast<const uint64_t*>(p1);
        const uint64_t* const p2_64 = reinterpret_cast<const uint64_t*>(p2);

        for (size_t i = 0; i < n / 8; ++i) {
            if (p1_64[i] != p2_64[i]) {
                return false;
            }
        }

        const uint8_t* const p1_8 = reinterpret_cast<const uint8_t*>(p1);
        const uint8_t* const p2_8 = reinterpret_cast<const uint8_t*>(p2);

        for (size_t i = n - n % 8; i < n; ++i) {
            if (p1_8[i] != p2_8[i]) {
                return false;
            }
        }

        return true;
    }

    __host__ __device__ void copy_mem(void* const dst, const void* const src, const size_t n) {
        // Kopiowanie bloków po 8 bajtów, potem kopiowanie pozostałych bajt po bajcie.
        uint64_t* const dst_64 = reinterpret_cast<uint64_t*>(dst);
        const uint64_t* const src_64 = reinterpret_cast<const uint64_t*>(src);

        for (size_t i = 0; i < n / 8; ++i) {
            dst_64[i] = src_64[i];
        }

        uint8_t* const dst_8 = reinterpret_cast<uint8_t*>(dst);
        const uint8_t* const src_8 = reinterpret_cast<const uint8_t*>(src);

        for (size_t i = n - n % 8; i < n; ++i) {
            dst_8[i] = src_8[i];
        }
    }

    __host__ __device__ void swap_mem(void* const p1, void* const p2, const size_t n) {
        // Swap bloków po 8 bajtów, potem swap pozostałych bajt po bajcie
        uint64_t* const p1_64 = reinterpret_cast<uint64_t*>(p1);
        uint64_t* const p2_64 = reinterpret_cast<uint64_t*>(p2);

        for (size_t i = 0; i < n / 8; ++i) {
            uint64_t temp = p1_64[i];
            p1_64[i] = p2_64[i];
            p2_64[i] = temp;
        }

        uint8_t* const p1_8 = reinterpret_cast<uint8_t*>(p1);
        uint8_t* const p2_8 = reinterpret_cast<uint8_t*>(p2);

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
