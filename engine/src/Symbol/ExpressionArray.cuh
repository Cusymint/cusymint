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
        // Maksymalna liczba symboli w pojedynczym wyrażeniu
        size_t expression_size = 0;
        // Maksymalna liczba wyrażeń w tablicy
        size_t expression_capacity = 0;
        // Obecna liczba wyrażeń w tablicy
        size_t expression_count = 0;

        template <typename U> friend class ExpressionArray;

      public:
        /*
         * @brief Tworzy tablicę `expression_count` wyrażeń z których każde ma długość maksymalną
         * `expression_size`
         */
        ExpressionArray(const size_t expression_capacity, const size_t expression_size,
                        const size_t expression_count = 0) :
            expression_size(expression_size),
            expression_capacity(expression_capacity),
            expression_count(expression_count),
            data(expression_size * expression_capacity) {}

        template <class U>
        ExpressionArray(const ExpressionArray<U>& other) // NOLINT(google-explicit-constructor)
            :
            data(other.data),
            expression_size(other.expression_size),
            expression_capacity(other.expression_capacity),
            expression_count(other.expression_count) {}

        template <class U>
        ExpressionArray(ExpressionArray<U>&& other) // NOLINT(google-explicit-constructor)
            :
            data(std::forward(other.data)),
            expression_size(other.expression_size),
            expression_capacity(other.expression_capacity),
            expression_count(other.expression_count) {}

        template <class U> ExpressionArray& operator=(const ExpressionArray<U>& other) {
            data = other.data;
            expression_size = other.expression_size;
            expression_count = other.expression_count;
            expression_capacity = other.expression_capacity;
            return *this;
        }

        ExpressionArray(const std::vector<std::vector<Symbol>>& expressions,
                        const size_t expression_size, const size_t expression_capacity) :
            expression_size(expression_size),
            expression_capacity(expression_capacity),
            expression_count(expressions.size()),
            data(expression_size * expression_capacity) {
            for (size_t expr_idx = 0; expr_idx < expressions.size(); ++expr_idx) {
                cudaMemcpy(data.at(expression_size * expr_idx), expressions[expr_idx].data(),
                           expressions[expr_idx].size() * sizeof(Sym::Symbol),
                           cudaMemcpyHostToDevice);
            }
        }

        /*
         * @brief Kopiuje dane tablicy do vectora vectorów
         */
        std::vector<std::vector<Symbol>> to_vector() const {
            std::vector<std::vector<Symbol>> expressions;
            expressions.reserve(expression_count);

            for (size_t expr_idx = 0; expr_idx < expression_count; ++expr_idx) {
                expressions.emplace_back(to_vector(expr_idx));
            }

            return expressions;
        }

        /*
         * @brief Kopiuje idx-te wyrażenie do tablicy vectorów
         */
        std::vector<Symbol> to_vector(const size_t idx) const {
            std::vector<Symbol> expression(expression_size);

            cudaMemcpy(expression.data(), (*this)[idx], expression_size * sizeof(Symbol),
                       cudaMemcpyDeviceToHost);
            expression.resize(expression.front().size());
            return expression;
        }

        /*
         * @brief Liczba ciągów w tablicy
         */
        __host__ __device__ size_t size() const { return expression_count; }

        /*
         * @brief Pojemność tablicy
         */
        __host__ __device__ size_t capacity() const { return expression_capacity; }

        /*
         * @brief Zmienia liczbę wyrażeń.
         *
         * @param new_expression_count Nowa liczba wyrażeń. Musi zachodzić `new_expression_count <=
         * capacity()`.
         */
        __host__ __device__ inline void resize(const size_t new_expression_count) {
            expression_count = new_expression_count;
        }

        /*
         * @brief Zmienia liczbę wyrażeń, którą pobiera z GPU
         *
         * @param new_expression_count Wskaźnik do nowej liczby wyrażeń.
         */
        inline void resize_from_device(const size_t* const new_expression_count) {
            cudaMemcpy(&expression_count, new_expression_count, sizeof(size_t),
                       cudaMemcpyDeviceToHost);
        }

        /*
         * @brief Dodaje wartość z GPU do liczby wyrażeń
         *
         * @param expression_count_increment Wskaźnik do nowej liczby wyrażeń.
         */
        void increment_size_from_device(const size_t* const expression_count_increment) {
            size_t increment = 0;
            cudaMemcpy(&increment, expression_count_increment, sizeof(size_t),
                       cudaMemcpyDeviceToHost);
            expression_count += increment;
        }

        /*
         * @brief Maksymalny rozmiar wyrażenia
         */
        __host__ __device__ size_t expression_max_size() const { return expression_size; }

        /*
         * @brief Wskaźnik do idx-tego ciągu symboli
         */
        __host__ __device__ const T* operator[](const size_t idx) const {
            return data.at(expression_size * idx)->as_ptr<T>();
        }

        /*
         * @brief Wskaźnik do idx-tego ciągu symboli
         */
        __host__ __device__ T* operator[](const size_t idx) {
            return const_cast<T*>((*const_cast<const ExpressionArray<T>*>(this))[idx]);
        }
    };
}

#endif
