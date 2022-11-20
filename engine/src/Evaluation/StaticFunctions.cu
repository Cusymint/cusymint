#include "StaticFunctions.cuh"

#include "Utils/DeviceArray.cuh"
#include "Utils/Meta.cuh"

#include "Symbol/MetaOperators.cuh"

namespace Sym::Static {
    namespace {
        using StaticFunctionInitializer = void (*)(Symbol&);

        __device__ Symbol IDENTITY[1];
        __device__ Symbol SIN_X[2];
        __device__ Symbol COS_X[2];
        __device__ Symbol E_TO_X[3];
        __device__ Symbol* const STATIC_FUNCTIONS[] = {
            IDENTITY,
            SIN_X,
            COS_X,
            E_TO_X,
        };

        __device__ void init_identity(Symbol& dst) { Var::init(dst); }
        __device__ void init_sin_x(Symbol& dst) { Sin<Var>::init(dst); }
        __device__ void init_cos_x(Symbol& dst) { Cos<Var>::init(dst); }
        __device__ void init_e_to_x(Symbol& dst) { Pow<E, Var>::init(dst); }

        __device__ const StaticFunctionInitializer STATIC_FUNCTIONS_INITIALIZERS[] = {
            init_identity,
            init_sin_x,
            init_cos_x,
            init_e_to_x,
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
