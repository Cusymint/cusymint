#ifndef OPTIONAL_NUMBER_H
#define OPTIONAL_NUMBER_H

#include "CompileConstants.cuh"
#include "Cuda.cuh"
#include <sys/types.h>

namespace Util {
    struct EmptyOptionalNumber {};

    const EmptyOptionalNumber empty_num;

    template <class T> struct OptionalNumber {
      private:
        bool _has_value = true;
        T _value;

      public:
        __host__ __device__ OptionalNumber() : _has_value(false), _value(T()) {}
        __host__ __device__ OptionalNumber(EmptyOptionalNumber /*empty*/) :
            _has_value(false), _value(T()) {}
        __host__ __device__ OptionalNumber(T number) : _value(number) {}
        __host__ __device__ OptionalNumber(OptionalNumber<T>& other) :
            _has_value(other._has_value), _value(other._value) {}
        __host__ __device__ OptionalNumber(OptionalNumber<T>&& other) noexcept :
            _has_value(other._has_value), _value(other._value) {}

        ~OptionalNumber() = default;

        __host__ __device__ inline bool has_value() const { return _has_value; }
        __host__ __device__ inline T value() const {
            if constexpr (Consts::DEBUG) {
                if (!_has_value) {
                    Util::crash("Trying to access a value of an empty OptionalNumber");
                }
            }
            return _value;
        }

        __host__ __device__ OptionalNumber<T>& operator=(const OptionalNumber<T>& other) {
            if (&other != this) {
                _has_value = other._has_value;
                _value = other._value;
            }
            return *this;
        }

        OptionalNumber<T>& operator=(OptionalNumber<T>&& other) noexcept = default;
    };

    template <class T>
    __host__ __device__ static OptionalNumber<T> operator+(const OptionalNumber<T>& num1,
                                                           const OptionalNumber<T>& num2) {
        if (num1.has_value() && num2.has_value()) {
            return num1.value() + num2.value();
        }
        return empty_num;
    }

    template <class T>
    __host__ __device__ static OptionalNumber<T> operator-(const OptionalNumber<T>& num1,
                                                           const OptionalNumber<T>& num2) {
        if (num1.has_value() && num2.has_value()) {
            return num1.value() - num2.value();
        }
        return empty_num;
    }

    template <class T>
    __host__ __device__ static OptionalNumber<T> operator*(const OptionalNumber<T>& num1,
                                                           const OptionalNumber<T>& num2) {
        if (num1.has_value() && num2.has_value()) {
            return num1.value() * num2.value();
        }
        return empty_num;
    }

    template <class T>
    __host__ __device__ static OptionalNumber<T> operator/(const OptionalNumber<T>& num1,
                                                           const OptionalNumber<T>& num2) {
        if (num1.has_value() && num2.has_value()) {
            return num1.value() / num2.value();
        }
        return empty_num;
    }

    template <class T>
    __host__ __device__ static OptionalNumber<T> operator-(const OptionalNumber<T>& num) {
        if (num.has_value()) {
            return -num.value();
        }
        return empty_num;
    }

    template <class T>
    __host__ __device__ static OptionalNumber<T> max(const OptionalNumber<T>& num1,
                                                     const OptionalNumber<T>& num2) {
        if (num1.has_value() && num2.has_value()) {
            return num1.value() < num2.value() ? num2.value() : num1.value();
        }
        return empty_num;
    }

}

#endif