#include "StaticFunctions.cuh"

#include "Utils/DeviceArray.cuh"
#include "Utils/Meta.cuh"

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

        __device__ void init_identity(Symbol& dst) { dst.init_from(Variable::create()); }

        __device__ void init_sin_x(Symbol& dst) {
            Sine* const sine = dst << Sine::builder();
            sine->arg().variable = Variable::create();
            sine->seal();
        }

        __device__ void init_cos_x(Symbol& dst) {
            Cosine* const cosine = dst << Cosine::builder();
            cosine->arg().variable = Variable::create();
            cosine->seal();
        }

        __device__ void init_e_to_x(Symbol& dst) {
            Power* const power = dst << Power::builder();
            power->arg1().known_constant = KnownConstant::with_value(KnownConstantValue::E);
            power->seal_arg1();
            power->arg2().variable = Variable::create();
            power->seal();
        }

        __device__ const StaticFunctionInitializer STATIC_FUNCTIONS_INITIALIZERS[] = {
            init_identity,
            init_sin_x,
            init_cos_x,
            init_e_to_x,
        };

        constexpr size_t STATIC_FUNCTION_COUNT =
            Util::ensure_same<Util::array_len(STATIC_FUNCTIONS),
                              Util::array_len(STATIC_FUNCTIONS_INITIALIZERS)>();

        __global__ void init_static_functions_kernel() {
            const size_t thread_idx = Util::thread_idx();
            const size_t thread_count = Util::thread_count();

            for (size_t i = thread_idx; i < STATIC_FUNCTION_COUNT; i += thread_count) {
                STATIC_FUNCTIONS_INITIALIZERS[i](*STATIC_FUNCTIONS[i]);
            }
        };
    };

    __device__ const Symbol* identity() { return IDENTITY; }

    __device__ const Symbol* sin_x() { return SIN_X; }

    __device__ const Symbol* cos_x() { return COS_X; }

    __device__ const Symbol* e_to_x() { return E_TO_X; }

    void init_functions() {
        static constexpr size_t BLOCK_SIZE = 1024;
        static constexpr size_t BLOCK_COUNT = 1;
        init_static_functions_kernel<<<BLOCK_COUNT, BLOCK_SIZE>>>();
    }
}
