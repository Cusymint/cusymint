#ifndef EXPRESSION_ARRAY_CUH
#define EXPRESSION_ARRAY_CUH

#include "Symbol.cuh"

#include "Utils/DeviceArray.cuh"

namespace Sym {
    /*
     * @brief Tablica ciągów symboli o stałej maksymalnej długości z których każdy zaczyna się
     * symbolem typu T
     */
    template <class T = Symbol> class ExpressionArray {
        Util::DeviceArray<Symbol> data;
        size_t expression_size = 0;
        size_t expression_count = 0;
        size_t expression_capacity = 0;

        template <typename U> friend class ExpressionArray;

      public:
        /*
         * @brief Tworzy tablicę `expression_count` wyrażeń z których każde ma długość maksymalną
         * `expression_size`
         */
        ExpressionArray(const size_t expression_capacity, const size_t expression_size) :
            expression_size(expression_size),
            expression_capacity(expression_capacity),
            data(expression_size * expression_capacity) {}

        template <class U>
        ExpressionArray(const ExpressionArray<U>& other) // NOLINT(google-explicit-constructor)
            :
            data(other.data),
            expression_size(other.expression_size),
            expression_count(other.expression_count),
            expression_capacity(other.expression_capacity) {}

        template <class U> ExpressionArray& operator=(const ExpressionArray<U>& other) {
            data = other.data;
            expression_size = other.expression_size;
            expression_count = other.expression_count;
            expression_capacity = other.expression_capacity;
            return *this;
        }

        /*
         * @brief Zwraca liczbę ciągów w tablicy
         */
        __host__ __device__ size_t size() const { return expression_count; }

        /*
         * @brief Zwraca pojemność tablicy
         */
        __host__ __device__ size_t capacity() const { return expression_capacity; }

        /*
         * @brief Zwraca maksymalny rozmiar wyrażenia
         */
        __host__ __device__ size_t expression_max_size() const { return expression_size; }

        /*
         * @brief Zwraca wskaźnik do idx-tego ciągu symboli
         */
        __device__ const T* operator[](const size_t idx) const {
            return &data[expression_size * idx].as<T>();
        }

        /*
         * @brief Zwraca wskaźnik do idx-tego ciągu symboli
         */
        __device__ T* operator[](const size_t idx) { return const_cast<T*>((*this)[idx]); }
    };
}

#endif
