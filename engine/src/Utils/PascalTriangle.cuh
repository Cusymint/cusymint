#include "CompileConstants.cuh"
#include "Symbol/Macros.cuh"
#include "Utils/Cuda.cuh"

namespace Util {
    struct PascalTriangle {
        size_t* data;
        size_t size = Sym::BUILDER_SIZE;

        __host__ __device__ static PascalTriangle generate(size_t n, size_t* data) {
            for (size_t i = 0; i <= n; ++i) {
                const auto offset = i * (i + 1) / 2;
                data[offset] = 1;
                data[offset + i] = 1;
                for (size_t j = 1; j < i; ++j) {
                    const auto off2 = offset + j;
                    data[off2] = data[off2 - i] + data[off2 - i - 1];
                }
            }
            return {data, n};
        }

        __host__ __device__ size_t binom(size_t n, size_t i) const {
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

        __host__ __device__ size_t occupied_size() const { return (size + 1) * (size + 2) / 2; }
    };
}