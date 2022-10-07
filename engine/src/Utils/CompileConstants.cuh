#ifndef COMPILE_CONSTANTS_CUH
#define COMPILE_CONSTANTS_CUH

namespace Util {
    static constexpr bool DEBUG =
#ifdef __DEBUG__
        true
#else
        false
#endif
        ;
}

#endif
