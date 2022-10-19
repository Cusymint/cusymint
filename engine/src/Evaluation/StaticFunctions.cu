#include "StaticFunctions.cuh"

#include "Utils/DeviceArray.cuh"
#include "Utils/Meta.cuh"

using StaticFunctionInitializer = void (*)(Sym::Symbol* const);

namespace {
    __device__ Sym::Symbol SIN_X[2];
    __device__ Sym::Symbol COS_X[2];
    __device__ Sym::Symbol E_TO_X[3];
    __device__ Sym::Symbol* const STATIC_FUNCTIONS[] = {
        SIN_X,
        COS_X,
        E_TO_X,
    };

    __device__ void init_sin_x(Sym::Symbol* const dst) {
        Sym::Sine* const sine = dst << Sym::Sine::builder();
        sine->arg().variable = Sym::Variable::create();
        sine->seal();
    }

    __device__ void init_cos_x(Sym::Symbol* const dst) {

        Sym::Cosine* const cosine = dst << Sym::Cosine::builder();
        cosine->arg().variable = Sym::Variable::create();
        cosine->seal();
    }

    __device__ void init_e_to_x(Sym::Symbol* const dst) {
        Sym::Power* const power = dst << Sym::Power::builder();
        power->arg1().known_constant = Sym::KnownConstant::with_value(Sym::KnownConstantValue::E);
        power->seal_arg1();
        power->arg2().variable = Sym::Variable::create();
        power->seal();
    }

    __device__ const StaticFunctionInitializer STATIC_FUNCTIONS_INITIALIZERS[] = {
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
            STATIC_FUNCTIONS_INITIALIZERS[i](STATIC_FUNCTIONS[i]);
        }
    };
}

namespace Sym {
    __device__ const Symbol* sin_x() { return SIN_X; }

    __device__ const Symbol* cos_x() { return COS_X; }

    __device__ const Symbol* e_to_x() { return E_TO_X; }

    void init_static_functions() {
        static constexpr size_t BLOCK_SIZE = 1024;
        static constexpr size_t BLOCK_COUNT = 1;
        init_static_functions_kernel<<<BLOCK_COUNT, BLOCK_SIZE>>>();
    }
}
