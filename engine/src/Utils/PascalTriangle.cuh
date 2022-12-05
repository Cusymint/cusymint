#ifndef PASCAL_TRIANGLE_CUH
#define PASCAL_TRIANGLE_CUH

#include "CompileConstants.cuh"
#include "Symbol/Macros.cuh"
#include "Utils/Cuda.cuh"

namespace Util {
    /*
     * @brief Structure used to calculate binomial coefficient values.
     * Values are stored in one-dimensional array in such a way:
     * binom(0,0) | binom(1,0) | binom(1,1) | binom(2,0) | binom(2,1) | binom(2,2) | ...
     */
    struct PascalTriangle {
        size_t* const data;
        const size_t size = Sym::BUILDER_SIZE;

        /*
         * @brief Generates a Pascal's triangle of height `n` in array pointed to by `data`.
         * `data`'s length must be equal or greater than `occupied_size()`.
         * This is the only proper way to obtain `PascalTriangle`.
         *
         * @param `n` Height of Pascal's triangle to be generated.
         * @param `data` An array to fill with generated values.
         *
         * @return A `PascalTriangle` structure representing generated triangle.
         */
        __host__ __device__ static PascalTriangle generate(const size_t n, size_t& data) {
            size_t* const data_ptr = &data;
            for (size_t i = 0; i <= n; ++i) {
                const auto offset = i * (i + 1) / 2;
                data_ptr[offset] = 1;
                data_ptr[offset + i] = 1;
                for (size_t j = 1; j < i; ++j) {
                    const auto off2 = offset + j;
                    data_ptr[off2] = data_ptr[off2 - i] + data_ptr[off2 - i - 1];
                }
            }
            return {data_ptr, n};
        }

        /*
         * @brief Retrieves previously generated value of binomial coefficient 'n over i'.
         * `n` must not be larger than triangle's `size` and `i` must be at most `n`.
         * `this` needs to be prevoiusly obtained by function `generate()`.
         *
         * @param `n` Upper number of binomial symbol
         * @param `i` Lower number of binomial symbol
         *
         * @return Binomial coefficient 'n over i' value.
         */
        __host__ __device__ size_t binom(const size_t n, const size_t i) const {
            if constexpr (Consts::DEBUG) {
                if (n < i || n > size) {
                    Util::crash("Trying to get binomial value for improper data: n=%lu, i=%lu, "
                                "triangle size=%lu",
                                n, i, size);
                }
                if (n == Sym::BUILDER_SIZE) {
                    Util::crash("Trying to get a value for not generated() triangle");
                }
            }
            return data[n * (n + 1) / 2 + i];
        }

        /*
         * @brief Returns number of values stored in `data`.
         *
         * @return As above.
         */
        __host__ __device__ size_t occupied_size() const { return (size + 1) * (size + 2) / 2; }
    };
}

#endif