#ifndef ORDER_CUH
#define ORDER_CUH

namespace Util {
    /*
     * @brief Enumeration that can be used as result of comparisons
     */
    enum class Order {
        Equal,
        Less,
        Greater,
    };

    template <class T> __host__ __device__ Order compare(const T& first, const T& second) {
        if (first < second) {
            return Order::Less;
        }

        if (first > second) {
            return Order::Greater;
        }

        return Order::Equal;
    }
}

#endif
