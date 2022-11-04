#ifndef STATIC_STACK_H
#define STATIC_STACK_H

#include "Utils/CompileConstants.cuh"
#include "Utils/Cuda.cuh"
namespace Util {
    template <class T> class StaticStack {
        T* data;
        size_t size;

        public:
            /*
             * @brief Creates a `StaticStack` structure, which operates on already allocated memory pointed to by `mem`.
             */
            __host__ __device__ explicit StaticStack(void* mem): data(reinterpret_cast<T*>(mem)), size(0) { }

            __host__ __device__ inline void push(T val) {
                data[size++] = val;
            }

            __host__ __device__ inline T pop() {
                if constexpr (Consts::DEBUG) {
                    if (empty()) {
                        crash("Trying to pop element from an empty StaticStack");
                    }
                }

                return data[--size];
            }

            __host__ __device__ inline size_t count() {
                return size;
            }

            __host__ __device__ inline bool empty() {
                return size == 0;
            }
    };
}

#endif