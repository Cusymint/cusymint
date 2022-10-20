#ifndef META_CUH
#define META_CUH

namespace Util {
    template <class T, size_t N> constexpr size_t array_len(const T (&array)[N]) noexcept {
        return N;
    }

    template <auto first, decltype(first) second> constexpr auto ensure_same() noexcept {
        static_assert(std::is_integral_v<decltype(first)>);
        static_assert(first == second, "Different values provided to ensure_same");
        return first;
    }
}

#endif
