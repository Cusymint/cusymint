#ifndef STATIC_STACK_H
#define STATIC_STACK_H

namespace Util {
    template <class T> class StaticStack {
        T* table;
        size_t current_idx;

        public:
            __host__ __device__ explicit StaticStack(void* table): table(reinterpret_cast<T*>(table)), current_idx(0) { }

            __host__ __device__ inline void push(T val) {
                table[current_idx++] = val;
            }

            __host__ __device__ inline T pop() {
                return table[--current_idx];
            }

            __host__ __device__ inline size_t count() {
                return current_idx;
            }

            __host__ __device__ inline bool empty() {
                return current_idx == 0;
            }
    };
}

#endif