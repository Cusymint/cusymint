#ifndef PAIR_CUH
#define PAIR_CUH

namespace Util {
    /*
     * @brief Helper class similar to `std::pair`. This one can be used with CUDA.
     */
    template <class T, class U> struct Pair {
        T first;
        U second;
    };
}

#endif
