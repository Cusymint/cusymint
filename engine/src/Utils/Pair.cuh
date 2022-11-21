#ifndef PAIR_CUH
#define PAIR_CUH

namespace Util {
    /*
     * @brief Helper class similar to `std::pair`. This one can be used with CUDA.
     */
    template <class T, class U> struct Pair {
        __host__ __device__ Pair(T&& first, U&& second) :
            first(std::move(first)), second(std::move(second)) {}

        T first;
        U second;
    };
}

#endif
