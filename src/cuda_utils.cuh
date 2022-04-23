#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

namespace Util {
    __device__ inline size_t thread_count() { return gridDim.x * blockDim.x; }
    __device__ inline size_t thread_idx() { return threadIdx.x + blockDim.x * blockIdx.x; }
}

#endif
