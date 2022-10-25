#include "StaticFunctions.cuh"

#include "Utils/DeviceArray.cuh"
#include "Utils/Meta.cuh"

namespace Sym::Static {
    namespace {
        using StaticFunctionInitializer = void (*)(Symbol&);

        __device__ Symbol IDENTITY[1];

        __device__ Symbol SIN_X[2];
        __device__ Symbol COS_X[2];
        __device__ Symbol TAN_X[2];
        __device__ Symbol COT_X[2];

        __device__ Symbol UNIVERSAL_SIN_X[2];
        __device__ Symbol UNIVERSAL_COS_X[2];
        __device__ Symbol UNIVERSAL_TAN_X[2];
        __device__ Symbol UNIVERSAL_COT_X[2];
        __device__ Symbol UNIVERSAL_DERIVATIVE[2];

        __device__ Symbol E_TO_X[3];

        __device__ Symbol* const STATIC_FUNCTIONS[] = {
            IDENTITY,        SIN_X,
            COS_X,           TAN_X,
            COT_X,           UNIVERSAL_SIN_X,
            UNIVERSAL_COS_X, UNIVERSAL_TAN_X,
            UNIVERSAL_COT_X, UNIVERSAL_DERIVATIVE,
            E_TO_X,
        };

        __device__ void init_identity(Symbol& dst) { dst.init_from(Variable::create()); }

        template <class T> __device__ void init_one_arg_function(Symbol& dst) {
            T* const sine = dst << T::builder();
            sine->arg().variable = Variable::create();
            sine->seal();
        }

        __device__ void init_sin_x(Symbol& dst) { init_one_arg_function<Sine>(dst); }
        __device__ void init_cos_x(Symbol& dst) { init_one_arg_function<Cosine>(dst); }
        __device__ void init_tan_x(Symbol& dst) { init_one_arg_function<Tangent>(dst); }
        __device__ void init_cot_x(Symbol& dst) { init_one_arg_function<Cotangent>(dst); }

        __device__ void init_1_plus_x2(Symbol& dst) {
            Addition* const addition = dst << Addition::builder();

            addition->arg1().init_from(NumericConstant::with_value(1.0));
            addition->seal_arg1();

            Power* const power = addition->arg2() << Power::builder();
            power->arg1().init_from(Variable::create());
            power->seal_arg1();
            power->arg2().init_from(NumericConstant::with_value(2.0));
            power->seal();

            addition->seal();
        }

        __device__ void init_1_minus_x2(Symbol& dst) {
            Addition* const addition = dst << Addition::builder();

            addition->arg1().init_from(NumericConstant::with_value(1.0));
            addition->seal_arg1();

            Negation* const negation = addition->arg2() << Negation::builder();

            Power* const power = negation->arg() << Power::builder();
            power->arg1().init_from(Variable::create());
            power->seal_arg1();
            power->arg2().init_from(NumericConstant::with_value(2.0));
            power->seal();

            negation->seal();

            addition->seal();
        }

        __device__ void init_2x(Symbol& dst) {
            Product* const product = dst << Product::builder();

            product->arg1().init_from(NumericConstant::with_value(2.0));
            product->seal_arg1();

            product->arg2().init_from(Variable::create());
            product->seal();
        }

        template <void (*I1)(Symbol&), void (*I2)(Symbol&)>
        __device__ void init_quotient(Symbol& dst) {
            Product* const product = dst << Product::builder();

            I1(product->arg1());
            product->seal_arg1();

            Reciprocal* const reciprocal = product->arg2() << Reciprocal::builder();
            I2(reciprocal->arg());
            reciprocal->seal();
            product->seal();
        }

        __device__ void init_universal_sin_x(Symbol& dst) {
            init_quotient<init_2x, init_1_plus_x2>(dst);
        }

        __device__ void init_universal_cos_x(Symbol& dst) {
            init_quotient<init_1_minus_x2, init_1_plus_x2>(dst);
        }

        __device__ void init_universal_tan_x(Symbol& dst) {
            init_quotient<init_2x, init_1_minus_x2>(dst);

            init_quotient<init_quotient<init_2x, init_2x>, init_1_minus_x2>(dst);
        }

        __device__ void init_universal_cot_x(Symbol& dst) {
            init_quotient<init_1_minus_x2, init_2x>(dst);
        }

        __device__ void init_universal_derivative(Symbol& dst) {
            Product* const product = dst << Product::builder();

            init_1_plus_x2(product->arg1());
            product->seal_arg1();

            product->arg2().init_from(NumericConstant::with_value(0.5));
            product->seal();
        }

        __device__ void init_e_to_x(Symbol& dst) {
            Power* const power = dst << Power::builder();
            power->arg1().known_constant = KnownConstant::with_value(KnownConstantValue::E);
            power->seal_arg1();
            power->arg2().variable = Variable::create();
            power->seal();
        }

        __device__ const StaticFunctionInitializer STATIC_FUNCTIONS_INITIALIZERS[] = {
            init_identity,        init_sin_x,
            init_cos_x,           init_tan_x,
            init_cot_x,           init_universal_sin_x,
            init_universal_cos_x, init_universal_tan_x,
            init_universal_cot_x, init_universal_derivative,
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
    __device__ const Symbol& tan_x() { return *TAN_X; }
    __device__ const Symbol& cot_x() { return *COT_X; }

    __device__ const Symbol& universal_sin_x() { return *UNIVERSAL_SIN_X; }
    __device__ const Symbol& universal_cos_x() { return *UNIVERSAL_COS_X; }
    __device__ const Symbol& universal_tan_x() { return *UNIVERSAL_TAN_X; }
    __device__ const Symbol& universal_cot_x() { return *UNIVERSAL_COT_X; }
    __device__ const Symbol& universal_derivative() { return *UNIVERSAL_DERIVATIVE; }

    __device__ const Symbol& e_to_x() { return *E_TO_X; }

    void init_functions() {
        static constexpr size_t BLOCK_SIZE = 1024;
        static constexpr size_t BLOCK_COUNT = 1;
        init_static_functions_kernel<<<BLOCK_COUNT, BLOCK_SIZE>>>();
    }
}
