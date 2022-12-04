#ifndef EXPRESSION_ARRAY_CUH
#define EXPRESSION_ARRAY_CUH

#include "Symbol.cuh"

#include "Utils/CompileConstants.cuh"
#include "Utils/DeviceArray.cuh"

namespace Sym {
    template <class T = Symbol>
    __global__ void set_new_expression_capacities(
        Util::DeviceArray<size_t> expression_capacities,
        Util::DeviceArray<size_t> expression_capacities_sum, const size_t old_expression_count,
        const size_t new_expression_count, const size_t new_expressions_capacity) {
        const size_t thread_idx = Util::thread_idx();
        const size_t thread_count = Util::thread_count();

        const size_t size_diff = new_expression_count - old_expression_count;
        const size_t old_expressions_total_capacity =
            old_expression_count == 0 ? 0 : expression_capacities_sum[old_expression_count - 1];

        for (size_t i = thread_idx; i < size_diff; i += thread_count) {
            expression_capacities[i + old_expression_count] = new_expressions_capacity;
            expression_capacities_sum[i + old_expression_count] =
                old_expressions_total_capacity + new_expressions_capacity * (i + 1);
        }
    }

    template <class T = Symbol>
    __global__ void reoffset_indices(const Util::DeviceArray<bool> indices,
                                     Util::DeviceArray<size_t> expression_capacities,
                                     Util::DeviceArray<size_t> expression_capacities_sum,
                                     const size_t realloc_multiplier) {
        const size_t thread_idx = Util::thread_idx();
        const size_t thread_count = Util::thread_count();

        const size_t indices_count = Util::min(indices.size(), expression_capacities.size());

        for (size_t i = thread_idx; i < indices_count; i += thread_count) {
            if (indices[i]) {
                expression_capacities[i] *= realloc_multiplier;
                // TODO: zmiany indeksów + przekopiowanie danych !!!!!!!!!!!
            }
        }
    }

    /*
     * @brief Array of expressions of different lengths in CUDA memory. Each expression begins with
     * a symbol of type T.
     */
    template <class T = Symbol> class ExpressionArray {
        // When doing a reallocation, how many times more memory to allocate than is actually needed
        static constexpr size_t REALLOC_MULTIPLIER = 2;

        Util::DeviceArray<Symbol> data;

        // Capacity of every expression. Can potentially have bigger size than there are
        // expressions. In that case, values starting with `expression_count`-th coordinate should
        // be treated as garbage.
        Util::DeviceArray<size_t> expression_capacities;

        // Cumulative sum of `expression_capacities`. Is always of the same size as
        // `expression_capacities`
        Util::DeviceArray<size_t> expression_capacities_sum;

        // Number of expressions actually held in the array
        size_t expression_count = 0;

        template <class U> friend class ExpressionArray;

      public:
        /*
         * @brief Creates an array with initial memory sizes
         *
         * @param symbols_capacity Initial number of `Symbols` that the array can hold without
         * reallocations
         * @param expressions_capacity Initial number of expressions that the array can hold without
         * reallocations
         */
        explicit ExpressionArray(const size_t symbols_capacity, const size_t expressions_capacity) :
            data(symbols_capacity),
            expression_capacities(expression_capacity),
            expression_capacities_sum(expressions_capacity) {}

        template <class U>
        ExpressionArray(const ExpressionArray<U>& other) // NOLINT(google-explicit-constructor)
            :
            data(other.data),
            expression_capacities(other.expression_capacities),
            expression_capacities_sum(other.expression_offsets),
            expression_count(expression_count) {}

        template <class U>
        ExpressionArray(ExpressionArray<U>&& other) // NOLINT(google-explicit-constructor)
            :
            data(std::forward(other.data)),
            expression_capacities(std::forward(other.expression_capacities)),
            expression_capacities_sum(std::forward(other.expression_offsets)),
            expression_count(other.expression_count) {}

        template <class U> ExpressionArray& operator=(const ExpressionArray<U>& other) {
            if (&other == this) {
                return *this;
            }

            data = other.data;
            expression_capacities = other.expression_capacities;
            expression_capacities_sum = other.expression_offsets;
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
            std::vector<size_t> expressions_sizes;
            std::vector<size_t> expressions_sizes_sum;

            size_t current_size_sum = 0;
            for (size_t i = 0; i < expression_count; ++i) {
                current_size_sum += expressions[i].size();
                expressions_sizes.push_back(expressions[i].size());
                expressions_sizes_sum.push_back(current_size_sum);
            }

            expression_capacities.resize(expressions.size());
            expression_capacities_sum.resize(expressions.size());

            cudaMemcpy(expression_capacities.data(), expressions_sizes.data(),
                       expressions_sizes.size() * sizeof(size_t), cudaMemcpyHostToDevice);
            cudaMemcpy(expression_capacities_sum.data(), expressions_sizes_sum.data(),
                       expressions_sizes_sum.size() * sizeof(size_t), cudaMemcpyHostToDevice);

            data.resize(current_size_sum);

            for (size_t expr_idx = 0; expr_idx < expressions.size(); ++expr_idx) {
                cudaMemcpy(at(expr_idx), expressions[expr_idx].data(),
                           expressions[expr_idx].size() * sizeof(Sym::Symbol),
                           cudaMemcpyHostToDevice);
            }

            expression_count = expressions.size();
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
         * @brief Changes the number of held expressions to `new_expression_count`.
         *
         * @param new_expression_count Number of expressions in the array after the call
         * @param new_expressions_capacity If new expressions will be created, they will all have a
         * capacity of `new_expressions_capacity`
         */
        void resize(const size_t new_expression_count, const size_t new_expressions_capacity = 1) {
            if (new_expression_count <= expression_count) {
                expression_count = new_expression_count;
                return;
            }

            const size_t additional_expr_count = new_expression_count - expression_count;

            if (expression_capacities.size() < new_expression_count) {
                expression_capacities.resize(new_expression_count * REALLOC_MULTIPLIER);
                expression_capacities_sum.resize(new_expression_count * REALLOC_MULTIPLIER);

                set_new_expression_capacities<<<1, 1024>>>(
                    expression_capacities, expression_capacities_sum, expression_count,
                    new_expression_count, new_expressions_capacity);
            }

            const size_t current_total_capacity =
                expression_count == 0 ? 0 : expression_capacities_sum.to_cpu(expression_count - 1);

            const size_t new_required_capacity =
                current_total_capacity + additional_expr_count * new_expressions_capacity;

            if (new_required_capacity > data.size()) {
                data.resize(new_required_capacity * REALLOC_MULTIPLIER);
            }

            expression_count = new_expression_count;
        }

        /*
         * @brief Multiplies capacities of expressions at selected indices by REALLOC_MULTIPLIER
         *
         * @param indices Array of `bool`s. If `indices[i] == true`, then the capacity of expression
         * at the `i`-th index will be multiplied by REALLOC_MULTIPLIER
         */
        void reoffset_indices(const Util::DeviceArray<bool> indices) {
            reoffset_indices<<<1, 1024>>>(indices, expression_capacities, expression_capacities_sum,
                                          REALLOC_MULTIPLIER);
        }

        /*
         * @brief Sets expression count and their offsets to the same as the ones in `other`.
         * Current contents of the array turns into garbage.
         *
         * @param other Array from which sizes and offsets are going to be copied
         */
        template <class U> void reoffset_like(ExpressionArray<U> other) {
            if (other.size() > data.size()) {
                data.resize(other.size() * REALLOC_MULTIPLIER);
            }

            if (other.expression_count > expression_capacities.size()) {
                expression_capacities.resize(other.expression_count * REALLOC_MULTIPLIER);
                expression_capacities_sum.resize(other.expression_count * REALLOC_MULTIPLIER);
            }

            cudaMemcpy(expression_capacities.data(), other.expression_capacities.data(),
                       other.expression_capacity * sizeof(size_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(expression_capacities_sum.data(), other.expression_capacities_sum.data(),
                       other.expression_capacity * sizeof(size_t), cudaMemcpyDeviceToDevice);

            expression_count = other.expression_count;
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

            return expression_capacities[idx];
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

            if (idx == 0) {
                return data.at(0);
            }

            return data.at(expression_capacities_sum[idx - 1]);
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
            [[nodiscard]] __host__ __device__ Symbol* operator->() const {
                return array->at(index_);
            }

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
    };
}

#endif
