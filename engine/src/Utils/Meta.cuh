#ifndef META_CUH
#define META_CUH

#include <cuda/std/tuple>

namespace Util {
    /*
     * @brief Structure sometimes useful in metaprogramming
     */
    struct Empty {};

    /*
     * @brief Returns length of a static array
     *
     * @tparam T Type of array elements (autodeduced)
     * @tparam N Number of elements of the array (autodeduced)
     *
     * @param Array to get the length from
     *
     * @return Length of the array
     */
    template <class T, size_t N>
    __host__ __device__ constexpr size_t array_len(const T (&/*array*/)[N]) noexcept {
        return N;
    }

    template <auto first, decltype(first) second> struct ensure_same {
        static_assert(std::is_integral_v<decltype(first)>);
        static_assert(first == second, "Different values provided to ensure_same");

        const static decltype(first) value = first;
    };

    /*
     * @brief Checks if `first` and `second` are equal, and if so is equal to their value
     *
     * @tparam first First value to compare
     * @tparam second Second value to compare
     */
    template <auto first, decltype(first) second>
    constexpr auto ensure_same_v = ensure_same<first, second>::value;

    template <size_t S, class Tuple, size_t... Is>
    __host__ __device__ constexpr auto _slice_tuple(const Tuple& tuple,
                                                    std::index_sequence<Is...> /*unused*/) {
        return cuda::std::make_tuple(cuda::std::get<S + Is>(tuple)...);
    }

    /*
     * @brief Extracts N parameters from a tuple starting at S
     *
     * @tparam S Index of the first tuple element to extract
     * @tparam N Number of tuple elements to extract
     * @tparam Tuple Type of tuple to extract elements from
     *
     * @param tuple Tuple to extract elements from
     *
     * @return A new tuple containing N elements
     */
    template <size_t S, size_t N, class Tuple>
    __host__ __device__ constexpr auto slice_tuple(const Tuple& tuple) {
        return _slice_tuple<S>(tuple, std::make_index_sequence<N>{});
    }

    template <class... Tuples>
    using TupleCat =
        typename std::invoke_result_t<decltype(cuda::std::tuple_cat<Tuples...>), Tuples...>;
}

#endif
