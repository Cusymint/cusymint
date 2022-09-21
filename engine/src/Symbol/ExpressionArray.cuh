#ifndef EXPRESSION_ARRAY_CUH
#define EXPRESSION_ARRAY_CUH

#include "Symbol.cuh"

#include "Utils/DeviceArray.cuh"

namespace Sym {
    /*
     * @brief Tablica ciągów symboli o stałej maksymalnej długości z których każdy zaczyna się
     * symbolem typu T
     */
    template <class T = Unknown> class ExpressionArray {
        Util::DeviceArray<Symbol> data;
        size_t expression_size;
        size_t expression_count;

      public:
        /*
         * @brief Tworzy tablicę `expression_count` wyrażeń z których każde ma długość maksymalną
         * `expression_size`
         */
        ExpressionArray(const size_t expression_count, const size_t expression_size) :
            expression_size(expression_size),
            expression_count(expression_count),
            data(expression_size * expression_size) {}

        /*
         * @brief Zwraca wskaźnik do idx-tego ciągu symboli
         */
        __device__ const T* operator[](const size_t idx) const {
            return &data[expression_size * idx];
        }

        /*
         * @brief Zwraca wskaźnik do idx-tego ciągu symboli
         */
        __device__ T* operator[](const size_t idx) {
            return const_cast<T*>(const_cast<const T>(*this)[idx]);
        }
    };
}

#endif
