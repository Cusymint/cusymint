#ifndef EXPRESSION_ARRAY_CUH
#define EXPRESSION_ARRAY_CUH

#include "Symbol.cuh"

#include "Utils/CompileConstants.cuh"
#include "Utils/DeviceArray.cuh"

namespace Sym {
    /*
     * @brief Array of expressions of different lengths in CUDA memory. Each expression begins with
     * a symbol of type T.
     */
    template <class T = Symbol> class ExpressionArray {
        Util::DeviceArray<Symbol> data;
        Util::DeviceArray<size_t> expression_offsets;

        // Number of expressions actually held in the array
        size_t expression_count = 0;

        template <typename U> friend class ExpressionArray;

      public:
        /*
         * @brief Creates an array with initial memory sizes
         *
         * @param capacity Initial size of data array (number of Symbols) holding expressions
         * @param expression_count Initial number of expressions in the array
         */
        ExpressionArray(const size_t capacity, const size_t expression_count) :
            data(capacity), expression_offsets(expression_count) {}

        template <class U>
        ExpressionArray(const ExpressionArray<U>& other) // NOLINT(google-explicit-constructor)
            :
            data(other.data),
            expression_offsets(other.expression_offsets),
            expression_count(expression_count) {}

        template <class U>
        ExpressionArray(ExpressionArray<U>&& other) // NOLINT(google-explicit-constructor)
            :
            data(std::forward(other.data)),
            expression_offsets(std::forward(other.expression_offsets)),
            expression_count(other.expression_count) {}

        template <class U> ExpressionArray& operator=(const ExpressionArray<U>& other) {
            if (&other == this) {
                return *this;
            }

            data = other.data;
            expression_offsets = other.expression_offsets;
            expression_count = other.expression_count;

            return *this;
        }

        /*
         * @brief Creates an array and copies over expressions from a vector of vectors
         */
        explicit ExpressionArray(const std::vector<std::vector<Symbol>>& expressions) {
            load_from_vector(expressions);
        }

        /*
         * @brief Reads all expressions from `expressions`, allocates memory for them and copies
         * them to this array. Leaves no spare capacity.
         */
        void load_from_vector(const std::vector<std::vector<Symbol>>& expressions) {
            expression_count = expressions.size();
            std::vector<size_t> offsets(expression_count);

            size_t current_offset = 0;
            for (size_t i = 0; i < expression_count; ++i) {
                offsets[i] = current_offset;
                current_offset += expressions[i].size();
            }

            expression_offsets.resize(expressions.size());
            cudaMemcpy(expression_offsets.data(), offsets.data(),
                       expressions.size() * sizeof(size_t), cudaMemcpyHostToDevice);

            data.resize(current_offset);

            for (size_t expr_idx = 0; expr_idx < expressions.size(); ++expr_idx) {
                cudaMemcpy(data.at(offsets[expr_idx]), expressions[expr_idx].data(),
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
                    if (index >= array.expression_count) {
                        Util::crash("Trying to create an iterator to an ExpressionArray past the "
                                    "last expression");
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
             * @brief Capacity of the expression pointed to by the iterator
             */
            [[nodiscard]] __host__ __device__ size_t capacity() const {
                return array->expression_capacity(index_);
            }

            /*
             * @brief Expression pointed to by the iterator
             */
            [[nodiscard]] __host__ __device__ T& operator*() const {
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
            [[nodiscard]] __host__ __device__ Symbol* operator->() const { return array->at(index_); }

            /*
             * @brief Pointer to the expression pointed to by the iterator
             */
            [[nodiscard]] __host__ __device__ const T* const_ptr() const { return &**this; }

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
        [[nodiscard]] __host__ __device__ Iterator iterator(const size_t index) {
            return Iterator(*this, index);
        }

        /*
         * @brief Constructs a const iterator pointing to the index-th expression
         */
        [[nodiscard]] __host__ __device__ ConstIterator const_iterator(const size_t index) const {
            return ConstIterator(*this, index);
        }

        /*
         * @brief Copies array data to a vector of vectors
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
         * @brief Copies the `idx`-th expression to a vector
         */
        [[nodiscard]] std::vector<Symbol> to_vector(const size_t idx) const {
            std::vector<Symbol> expression(expression_capacity(idx));

            cudaMemcpy(expression.data(), at(idx), expression_capacity(idx) * sizeof(Symbol),
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
         * @brief Number of expressions in the array
         */
        [[nodiscard]] __host__ __device__ size_t size() const { return expression_count; }

        /*
         * @brief Number of Symbols that can be stored in the memory currently allocated by the
         * array
         */
        [[nodiscard]] __host__ __device__ size_t symbols_capacity() const { return data.size(); }

        /*
         * @brief Number of expressions that can be stored by the array without reallocations
         */
        [[nodiscard]] __host__ __device__ size_t expressions_capacity() const {
            return expression_offsets.size();
        }

        /*
         * @brief Number of `Symbol`s that the `idx`-th expression can potentially hold without
         * changing offsets
         */
        [[nodiscard]] __host__ __device__ size_t expression_capacity(const size_t idx) const {
            if constexpr (Consts::DEBUG) {
                if (idx >= expression_count) {
                    Util::crash(
                        "Trying to access capacity of expression %lu of an ExpressionArray with "
                        "size of %lu",
                        idx, expression_count);
                }
            }

            if (idx == expression_count - 1) {
                return data.size() - expression_offsets[idx];
            }

            return expression_offsets[idx + 1] - expression_offsets[idx];
        }

        /*
         * @brief Const pointer to the `idx`-th expression
         */
        __host__ __device__ const T* at(const size_t idx) const {
            if constexpr (Consts::DEBUG) {
                if (idx >= expression_count) {
                    Util::crash("Trying to access expression %lu of an ExpressionArray with "
                                "size of %lu",
                                idx, expression_count);
                }
            }

            return data.at(expression_offsets[idx])->as_ptr<T>();
        }

        /*
         * @brief Pointer to the `idx`-th expression
         */
        __host__ __device__ T* at(const size_t idx) {
            return const_cast<T*>(const_cast<const ExpressionArray<T>*>(this)->at(idx));
        }

        /*
         * @brief Const reference to the `idx`-th expression
         */
        __host__ __device__ const T& operator[](const size_t idx) const { return *at(idx); }

        /*
         * @brief Reference to the `idx`-th expression
         */
        __host__ __device__ T& operator[](const size_t idx) {
            return const_cast<T&>((*const_cast<const ExpressionArray<T>*>(this))[idx]);
        }
    };
}

#endif
