#ifndef EXPRESSION_ARRAY_CUH
#define EXPRESSION_ARRAY_CUH

#include "Symbol.cuh"

#include "Utils/CompileConstants.cuh"
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
            if (&other == this) {
                return *this;
            }

            data = other.data;
            expression_size = other.expression_size;
            expression_count = other.expression_count;
            expression_capacity = other.expression_capacity;
            return *this;
        }

        ExpressionArray(const std::vector<std::vector<Symbol>>& expressions,
                        const size_t expression_size, const size_t expression_capacity) :
            expression_size(expression_size), expression_capacity(expression_capacity) {
            load_from_vector(expressions);
        }

        /*
         * @brief Reads all expressions from `expressions`, allocates memory for them and copies
         * them to this array
         */
        void load_from_vector(const std::vector<std::vector<Symbol>>& expressions) {
            expression_count = expressions.size();
            data.resize(expression_size * expression_capacity);

            for (size_t expr_idx = 0; expr_idx < expressions.size(); ++expr_idx) {
                cudaMemcpy(data.at(expression_size * expr_idx), expressions[expr_idx].data(),
                           expressions[expr_idx].size() * sizeof(Sym::Symbol),
                           cudaMemcpyHostToDevice);
            }
        }

        /*
         * @brief Iterator to an expression in an `ExpressionArray`
         */
        template <class U> class GenericIterator {
            U* array;
            size_t index_;

            __host__ __device__ GenericIterator() : array(nullptr), index_(0) {}

          public:
            __host__ __device__ GenericIterator(U& array, const size_t index) :
                array(&array), index_(index) {
                if constexpr (Consts::DEBUG) {
                    if (index > array.expression_capacity) {
                        Util::crash("Trying to create an iterator to an ExpressionArray past the "
                                    "allocated memory");
                    }
                }
            }

            __host__ __device__ static GenericIterator null() { return GenericIterator(); };

            [[nodiscard]] __host__ __device__ size_t index() const {
                if constexpr (Consts::DEBUG) {
                    if (array == nullptr) {
                        Util::crash("Trying to access the index of a null iterator");
                    }
                }

                return index_;
            }

            /*
             * @brief Expression pointed to by the iterator
             */
            __host__ __device__ T& operator*() const {
                if constexpr (Consts::DEBUG) {
                    if (array == nullptr) {
                        Util::crash("Trying to access a null iterator");
                    }
                }

                return array->operator[](index_);
            };

            /*
             * @brief Accesses the element pointed to by the iterator
             */
            __host__ __device__ Symbol* operator->() const { return array->at(index_); }

            /*
             * @brief Pointer to the expression pointed to by the iterator
             */
            [[nodiscard]] __host__ __device__ const T* const_ptr() const { return **this; }

            /*
             * @brief Pointer to the expression pointed to by the iterator
             */
            [[nodiscard]] __host__ __device__ T* ptr() const {
                return const_cast<Symbol*>(const_cast<const GenericIterator*>(this)->const_ptr());
            }

            /*
             * @brief Creates a new iterator offset by `offset` expressions
             */
            template <class O> __host__ __device__ GenericIterator operator+(const O offset) const {
                if constexpr (Consts::DEBUG) {
                    if (array == nullptr) {
                        Util::crash("Trying to offset a null iterator");
                    }
                }

                return GenericIterator(*array, index_ + offset);
            }

            /*
             * @brief Creates a new iterator offset by `-offset` expressions
             */
            template <class O> __host__ __device__ GenericIterator operator-(const O offset) const {
                return *this + (-offset);
            }

            /*
             * @brief Offsets the iterator by `offset`;
             */
            template <class O> __host__ __device__ GenericIterator operator+=(const O offset) {
                if constexpr (Consts::DEBUG) {
                    if (index_ + offset > array->expression_capacity() || -offset > index_) {
                        Util::crash("Trying to offset an iterator past owned memory (%lu + "
                                    "%l -> %lu)",
                                    index_, offset, index_ + offset);
                    }
                }

                index_ += offset;
                return *this;
            }

            /*
             * @brief Offsets the iterator by `-offset`;
             */
            template <class O> __host__ __device__ GenericIterator operator-=(const O offset) {
                return *this += -offset;
            }

            /*
             * @brief Offsets the iterator by `1`;
             */
            __host__ __device__ GenericIterator operator++() { return *this += 1; }

            /*
             * @brief Offsets the iterator by `-1`;
             */
            __host__ __device__ GenericIterator operator--() { return *this -= 1; }

            /*
             * @brief `index`-th symbol of the expression pointed to by the iterator
             */
            __host__ __device__ T& operator[](const size_t index) const { return array + index; }
        };

        using Iterator = GenericIterator<ExpressionArray>;
        using ConstIterator = GenericIterator<const ExpressionArray>;

        /*
         * @brief Constructs an iterator pointing to the index-th expression
         */
        __host__ __device__ Iterator iterator(const size_t index) { return Iterator(*this, index); }

        /*
         * @brief Constructs a const iterator pointing to the index-th expression
         */
        __host__ __device__ ConstIterator const_iterator(const size_t index) const {
            return ConstIterator(*this, index);
        }

        /*
         * @brief Kopiuje dane tablicy do vectora vectorów
         */
        [[nodiscard]] std::vector<std::vector<Symbol>> to_vector() const {
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
        [[nodiscard]] std::vector<Symbol> to_vector(const size_t idx) const {
            std::vector<Symbol> expression(expression_size);

            cudaMemcpy(expression.data(), at(idx), expression_size * sizeof(Symbol),
                       cudaMemcpyDeviceToHost);
            expression.resize(expression.front().size());
            return expression;
        }

        /*
         * @brief Copies the array onto the CPU and formats it as a string
         */
        [[nodiscard]] std::string to_string() const {
            const std::vector<std::vector<Symbol>> vec = to_vector();
            std::string string = "{\n";

            for (const auto& sym : vec) {
                string += sym.data()->to_string() + ",\n";
            }

            return string + "}";
        }

        /*
         * @brief Liczba ciągów w tablicy
         */
        [[nodiscard]] __host__ __device__ size_t size() const { return expression_count; }

        /*
         * @brief Pojemność tablicy
         */
        [[nodiscard]] __host__ __device__ size_t capacity() const { return expression_capacity; }

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
        template <class U> inline void resize_from_device(const U* const new_expression_count) {
            U new_count;
            cudaMemcpy(&new_count, new_expression_count, sizeof(U), cudaMemcpyDeviceToHost);
            expression_count = new_count;
        }

        /*
         * @brief Dodaje wartość z GPU do liczby wyrażeń
         *
         * @param expression_count_increment Wskaźnik do nowej liczby wyrażeń.
         */
        template <class U>
        void increment_size_from_device(const U* const expression_count_increment) {
            U increment;
            cudaMemcpy(&increment, expression_count_increment, sizeof(U), cudaMemcpyDeviceToHost);
            expression_count += increment;
        }

        /*
         * @brief Maksymalny rozmiar wyrażenia
         */
        [[nodiscard]] __host__ __device__ size_t expression_max_size() const {
            return expression_size;
        }

        /*
         * @brief Wskaźnik do idx-tego ciągu symboli
         */
        __host__ __device__ const T* at(const size_t idx) const {
            if constexpr (Consts::DEBUG) {
                if (idx > expression_capacity) {
                    Util::crash("Trying to access expression %lu of an ExpressionArray with "
                                "capacity of %lu",
                                idx, expression_capacity);
                }
            }

            return data.at(expression_size * idx)->as_ptr<T>();
        }

        /*
         * @brief Wskaźnik do idx-tego ciągu symboli
         */
        __host__ __device__ T* at(const size_t idx) {
            return const_cast<T*>(const_cast<const ExpressionArray<T>*>(this)->at(idx));
        }

        /*
         * @brief Wskaźnik do idx-tego ciągu symboli
         */
        __host__ __device__ const T& operator[](const size_t idx) const { return *at(idx); }

        /*
         * @brief Wskaźnik do idx-tego ciągu symboli
         */
        __host__ __device__ T& operator[](const size_t idx) {
            return const_cast<T&>((*const_cast<const ExpressionArray<T>*>(this))[idx]);
        }
    };
}

#endif
