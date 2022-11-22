#include "StaticFunctions.cuh"

#include "Utils/DeviceArray.cuh"
#include "Utils/Meta.cuh"

#include "Symbol/MetaOperators.cuh"

namespace Sym::Static {
    namespace {
        using StaticFunctionInitializer = void (*)(Symbol&);

        // NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables, cert-err58-cpp)
        __device__ Symbol IDENTITY[Var::Size::get_value()];

        using SinX = Sin<Var>;
        __device__ Symbol SIN_X[SinX::Size::get_value()];
        using CosX = Cos<Var>;
        __device__ Symbol COS_X[CosX::Size::get_value()];
        using TanX = Tan<Var>;
        __device__ Symbol TAN_X[TanX::Size::get_value()];
        using CotX = Cot<Var>;
        __device__ Symbol COT_X[CotX::Size::get_value()];

        using UniversalSinX = Frac<Mul<Integer<2>, Var>, Add<Integer<1>, Pow<Var, Integer<2>>>>;
        __device__ Symbol UNIVERSAL_SIN_X[UniversalSinX::Size::get_value()];

        using UniversalCosX =
            Frac<Sub<Integer<1>, Pow<Var, Integer<2>>>, Add<Integer<1>, Pow<Var, Integer<2>>>>;
        __device__ Symbol UNIVERSAL_COS_X[UniversalCosX::Size::get_value()];

        using UniversalTanX = Frac<Mul<Integer<2>, Var>, Sub<Integer<1>, Pow<Var, Integer<2>>>>;
        __device__ Symbol UNIVERSAL_TAN_X[UniversalTanX::Size::get_value()];

        using UniversalCotX = Frac<Sub<Integer<1>, Pow<Var, Integer<2>>>, Mul<Integer<2>, Var>>;
        __device__ Symbol UNIVERSAL_COT_X[UniversalCotX::Size::get_value()];

        using UniversalDerivative = Frac<Add<Integer<1>, Pow<Var, Integer<2>>>, Integer<2>>;
        __device__ Symbol UNIVERSAL_DERIVATIVE[UniversalDerivative::Size::get_value()];

        using TanXOver2 = Tan<Mul<Num, Var>>;
        __device__ Symbol TAN_X_OVER_2[TanXOver2::Size::get_value()];
        __device__ void init_tan_x_over_2(Symbol& dst) { TanXOver2::init(dst, {0.5}); }

        using EToX = Pow<E, Var>;
        __device__ Symbol E_TO_X[EToX::Size::get_value()];
        // NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables, cert-err58-cpp)

        __device__ Symbol* const STATIC_FUNCTIONS[] = {
            IDENTITY,        SIN_X,
            COS_X,           TAN_X,
            COT_X,           UNIVERSAL_SIN_X,
            UNIVERSAL_COS_X, UNIVERSAL_TAN_X,
            UNIVERSAL_COT_X, UNIVERSAL_DERIVATIVE,
            TAN_X_OVER_2,    E_TO_X,
        };

        __device__ const StaticFunctionInitializer STATIC_FUNCTIONS_INITIALIZERS[] = {
            Var::init,           SinX::init,
            CosX::init,          TanX::init,
            CotX::init,          UniversalSinX::init,
            UniversalCosX::init, UniversalTanX::init,
            UniversalCotX::init, UniversalDerivative::init,
            init_tan_x_over_2,   EToX::init,
        };

        constexpr size_t STATIC_FUNCTION_COUNT =
            Util::ensure_same_v<Util::array_len(STATIC_FUNCTIONS),
                                Util::array_len(STATIC_FUNCTIONS_INITIALIZERS)>;

        __global__ void init_static_functions_kernel() {
            const size_t thread_idx = Util::thread_idx();
            const size_t thread_count = Util::thread_count();

            for (size_t i = thread_idx; i < STATIC_FUNCTION_COUNT; i += thread_count) {
                STATIC_FUNCTIONS_INITIALIZERS[i](*STATIC_FUNCTIONS[i]);
            }
        };
    };

    __device__ const Symbol& identity() { return *IDENTITY; }
    __device__ const Symbol& sin_x() { return *SIN_X; }
    __device__ const Symbol& cos_x() { return *COS_X; }
    __device__ const Symbol& tan_x() { return *TAN_X; }
    __device__ const Symbol& cot_x() { return *COT_X; }

    __device__ const Symbol& universal_sin_x() { return *UNIVERSAL_SIN_X; }
    __device__ const Symbol& universal_cos_x() { return *UNIVERSAL_COS_X; }
    __device__ const Symbol& universal_tan_x() { return *UNIVERSAL_TAN_X; }
    __device__ const Symbol& universal_cot_x() { return *UNIVERSAL_COT_X; }
    __device__ const Symbol& universal_derivative() { return *UNIVERSAL_DERIVATIVE; }
    __device__ const Symbol& tan_x_over_2() { return *TAN_X_OVER_2; }

    __device__ const Symbol& e_to_x() { return *E_TO_X; }

    void init_functions() {
        static constexpr size_t BLOCK_SIZE = 1024;
        static constexpr size_t BLOCK_COUNT = 1;
        static bool are_initialized = false;

        if (!are_initialized) {
            init_static_functions_kernel<<<BLOCK_COUNT, BLOCK_SIZE>>>();
            are_initialized = true;
        }
    }
}
