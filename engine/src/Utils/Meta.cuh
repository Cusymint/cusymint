#ifndef META_CUH
#define META_CUH

#include <cuda/std/tuple>
#include <tuple>
#include <utility>

namespace Util {
    template <class... Optionals> struct MetaOptionalsSum;

    /*
     * @brief Like std::optional but everything is kept at compile time
     */
    template <auto V, bool H = true> class MetaOptional {
      public:
        using ValueType = decltype(V);

      private:
        static constexpr ValueType VALUE = V;

      public:
        static constexpr bool HAS_VALUE = H;

        static constexpr ValueType get_value() {
            static_assert(HAS_VALUE, "This MetaOptional is empty");
            return VALUE;
        };

        static constexpr ValueType get_value_unchecked() { return VALUE; };
    };

    template <class V> using MetaEmpty = MetaOptional<V{}, false>;

    template <class... Optionals>
    struct MetaOptionalsSum
        : MetaOptional<(Optionals::get_value_unchecked() + ...), (Optionals::HAS_VALUE && ...)> {};

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

    template <size_t N, size_t S, class Tuple, class T, size_t... LIs, size_t... RIs>
    __host__ __device__ constexpr auto
    _skip_n_and_insert_in_tuple_at(const Tuple& tuple, T& element,
                                   std::index_sequence<LIs...> /*left_indices*/,
                                   std::index_sequence<RIs...> /*right_indices*/) {
        return cuda::std::tie(cuda::std::get<LIs + N>(tuple)..., element,
                                     cuda::std::get<RIs + S + N>(tuple)...);
    }

    /*
     * @brief Skips N first elements in tuple, then inserts an element
     * in new tuple at position S (when S is equal to tuple size,
     * inserts the element at the end of a tuple).
     *
     * @tparam N Number of elements to skip in input tuple
     * @tparam S Index of an element to be inserted
     * @tparam Tuple Type of tuple to insert element to
     * @tparam T Type of inserted element
     *
     * @param tuple Tuple to insert element to
     * @param element Element to be inserted
     *
     * @return New tuple containing inserted element at S-th position.
     */
    template <size_t N, size_t S, class Tuple, class T>
    __host__ __device__ constexpr auto skip_n_and_insert_in_tuple_at(const Tuple& tuple,
                                                                     T& element) {
        return _skip_n_and_insert_in_tuple_at<N, S>(
            tuple, element, std::make_index_sequence<S>{},
            std::make_index_sequence<cuda::std::tuple_size_v<Tuple> - N - S>{});
    }

    template <size_t S, size_t N, class Tuple>
    using SliceTuple = typename std::invoke_result_t<decltype(slice_tuple<S, N, Tuple>), Tuple>;

    template <class... Tuples>
    using TupleCat =
        typename std::invoke_result_t<decltype(cuda::std::tuple_cat<Tuples...>), Tuples...>;

    template <class Tuple> constexpr bool is_empty_tuple = cuda::std::tuple_size_v<Tuple> == 0;
}

#endif
