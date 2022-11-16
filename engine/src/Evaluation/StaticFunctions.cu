#include "StaticFunctions.cuh"

#include "Utils/DeviceArray.cuh"
#include "Utils/Meta.cuh"

#include "Symbol/MetaOperators.cuh"

namespace Sym::Static {
    namespace {
        using StaticFunctionInitializer = void (*)(Symbol&);

        // NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
        __device__ Symbol IDENTITY[1];

        __device__ Symbol SIN_X[2];
        __device__ Symbol COS_X[2];
        __device__ Symbol TAN_X[2];
        __device__ Symbol COT_X[2];

        __device__ Symbol UNIVERSAL_SIN_X[9];
        __device__ Symbol UNIVERSAL_COS_X[11];
        __device__ Symbol UNIVERSAL_TAN_X[9];
        __device__ Symbol UNIVERSAL_COT_X[9];
        __device__ Symbol UNIVERSAL_DERIVATIVE[7];
        __device__ Symbol TAN_X_OVER_2[4];

        __device__ Symbol E_TO_X[3];
        // NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

        __device__ Symbol* const STATIC_FUNCTIONS[] = {
            IDENTITY,        SIN_X,
            COS_X,           TAN_X,
            COT_X,           UNIVERSAL_SIN_X,
            UNIVERSAL_COS_X, UNIVERSAL_TAN_X,
            UNIVERSAL_COT_X, UNIVERSAL_DERIVATIVE,
            TAN_X_OVER_2,    E_TO_X,
        };

        __device__ void init_identity(Symbol& dst) { Var::init(dst); }
        __device__ void init_sin_x(Symbol& dst) { Sin<Var>::init(dst); }
        __device__ void init_cos_x(Symbol& dst) { Cos<Var>::init(dst); }
        __device__ void init_tan_x(Symbol& dst) { Tan<Var>::init(dst); }
        __device__ void init_cot_x(Symbol& dst) { Cot<Var>::init(dst); }

        __device__ void init_universal_sin_x(Symbol& dst) {
            Frac<Prod<Integer<2>, Var>, Add<Integer<1>, Pow<Var, Integer<2>>>>::init(dst, {});
        }

        __device__ void init_universal_cos_x(Symbol& dst) {
            Frac<Sub<Integer<1>, Pow<Var, Integer<2>>>,
                 Add<Integer<1>, Pow<Var, Integer<2>>>>::init(dst, {});
        }

        __device__ void init_universal_tan_x(Symbol& dst) {
            Frac<Prod<Integer<2>, Var>, Sub<Integer<1>, Pow<Var, Integer<2>>>>::init(dst, {});
        }

        __device__ void init_universal_cot_x(Symbol& dst) {
            Frac<Sub<Integer<1>, Pow<Var, Integer<2>>>, Prod<Integer<2>, Var>>::init(dst, {});
        }

        __device__ void init_universal_derivative(Symbol& dst) {
            Frac<Add<Integer<1>, Pow<Var, Integer<2>>>, Integer<2>>::init(dst, {});
        }

        __device__ void init_tan_x_over_2(Symbol& dst) { Tan<Prod<Num, Var>>::init(dst, {0.5}); }

        __device__ void init_e_to_x(Symbol& dst) { Pow<E, Var>::init(dst); }

        __device__ const StaticFunctionInitializer STATIC_FUNCTIONS_INITIALIZERS[] = {
            init_identity,        init_sin_x,
            init_cos_x,           init_tan_x,
            init_cot_x,           init_universal_sin_x,
            init_universal_cos_x, init_universal_tan_x,
            init_universal_cot_x, init_universal_derivative,
            init_tan_x_over_2,    init_e_to_x,
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
        init_static_functions_kernel<<<BLOCK_COUNT, BLOCK_SIZE>>>();
    }
}
