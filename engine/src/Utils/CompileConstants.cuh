#ifndef COMPILE_CONSTANTS_CUH
#define COMPILE_CONSTANTS_CUH

namespace Consts {
    constexpr double EPS = 1e-10;

    static constexpr bool DEBUG =
#ifdef __DEBUG__
        true
#else
        false
#endif
        ;
}

#endif
