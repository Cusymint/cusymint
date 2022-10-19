#ifndef META_CUH
#define META_CUH

namespace Util {
    template <class T, size_t N> constexpr size_t array_len(const T (&array)[N]) { return N; }

    template <auto first, auto second> constexpr auto ensure_same() {
        static_assert(std::is_same_v<decltype(first), decltype(second)>);
        static_assert(std::is_integral_v<decltype(first)>);
        static_assert(first == second, "Different values provided to ensure_same");
        return first;
    }
}

#endif
