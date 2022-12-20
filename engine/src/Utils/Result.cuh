#ifndef RESULT_CUH
#define RESULT_CUH

#include "CompileConstants.cuh"
#include "Cuda.cuh"
#include "Meta.cuh"

#define TRY(_result)             \
    ({                           \
        auto result = (_result); \
        if (result.is_error()) { \
            return result;       \
        }                        \
                                 \
        result.good();           \
    })

#define TRY_PASS(_T, _result)         \
    ({                                \
        auto result = (_result);      \
        if (result.is_error()) {      \
            return result.pass<_T>(); \
        }                             \
                                      \
        result.good();                \
    })

namespace Util {
    /*
     * @brief Class for storing something, or an error
     */
    template <class T, class E> class Result {
        union ValueType {
            E error;
            T good;

            __host__ __device__ ValueType() : error() {}
        };

        ValueType value;
        bool is_good_ = false;

      public:
        /*
         * @brief Creates a result containing a constructed good result
         */
        template <class... Args> __host__ __device__ static Result make_good(Args... args) {
            Result result;
            result.value.good = T(args...);
            result.is_good_ = true;

            return result;
        };

        /*
         * @brief Creates a result containing a constructed error
         */
        template <class... Args> __host__ __device__ static Result make_error(Args... args) {
            Result result;
            result.value.error = E(args...);
            result.is_good_ = false;

            return result;
        };

        /*
         * @brief Checks if result contains a good value
         */
        [[nodiscard]] __host__ __device__ bool is_good() const { return is_good_; }

        /*
         * @brief Checks if result contains an error
         */
        [[nodiscard]] __host__ __device__ bool is_error() const { return !is_good_; }

        /*
         * @brief Extracts a good value out of the result. Undefined behaviour if actually contains
         * an error.
         */
        [[nodiscard]] __host__ __device__ T good() const {
            unwrap();
            return value.good;
        }

        /*
         * @brief Ignores the warning about unused result, crashes if it is an error
         */
        __host__ __device__ void unwrap() const {
            if constexpr (Consts::DEBUG) {
                if (is_error()) {
                    Util::crash("Trying to get a good value out of an error");
                }
            }
        }

        /*
         * @brief Extracts an error out of the result. Undefined behaviour if actually contains
         * a good value.
         */
        [[nodiscard]] __host__ __device__ E error() const {
            if constexpr (Consts::DEBUG) {
                if (is_good()) {
                    Util::crash("Trying to get an error out of a good value");
                }
            }

            return value.error;
        }

        /*
         * @brief If `T` == `E`, returns the contained value no matter what it is
         */
        [[nodiscard]] __host__ __device__ T any() {
            static_assert(std::is_same_v<T, E>,
                          "To use `any`, the good type has to be equal to the error type");

            return value.good;
        };

        /*
         * @brief The error contained in the result, but with a different type of good value
         */
        template <class U> [[nodiscard]] __host__ __device__ Result<U, E> pass() const {
            return Result<U, E>::make_error(error());
        }

        /*
         * @brief If contains a good value only changes the type. If contains an error, maps it to a
         * different error.
         */
        template <class F, class M>
        [[nodiscard]] __host__ __device__ Result<T, F> map_err(const M& mapper) const {
            if (is_good()) {
                return Result<T, F>::make_good(good());
            }

            Result<T, F>::make_error(mapper(error()));
        }

        /*
         * @brief If contains an error only changes the type. If contains a good value, maps it to a
         * different good value.
         */
        template <class U, class M>
        [[nodiscard]] __host__ __device__ Result<U, E> map_good(const M& mapper) const {
            if (is_error()) {
                return Result<U, E>::make_error(error());
            }

            Result<U, E>::make_good(mapper(good()));
        }
    };

    template <class T> using SimpleResult = Result<T, Empty>;
    using BinaryResult = Result<Empty, Empty>;
}

#endif
