#ifndef EXPRESSION_ARRAY_CUH
#define EXPRESSION_ARRAY_CUH

#include "Symbol.cuh"

#include <optional>

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "Evaluation/Status.cuh"
#include "Utils/CompileConstants.cuh"
#include "Utils/DeviceArray.cuh"

namespace Sym {
    namespace ExpressionArrayKernel {
        /*
         * @brief Expands expression capacity of an `ExpressionArray`
         *
         * @param expression_capacities Current capacities
         * @param expression_capacities_sum Current capacities cumulative sum
         * @param old_expression_count Current expression count
         * @param new_expression_count New expression count
         * @param new_expressions_capacity Capacity of each newly created expression
         */
        __global__ void set_new_expression_capacities(
            Util::DeviceArray<size_t> expression_capacities,
            Util::DeviceArray<size_t> expression_capacities_sum, const size_t old_expression_count,
            const size_t new_expression_count, const size_t new_expressions_capacity);

        /*
         * @brief Multiplies the capacities of `expression_capacities`
         *
         * @param statuses The `i`-th element of the array is multiplied only if
         * `statuses[i] == EvaluationStatus::ReallocationRequest`
         * @param expression_capacities Capacities to multiple
         * @param realloc_multiplier Constant by which to multiply the capacities
         * @param expression_count Number of expressions described by `expression_capacities`
         * @param start Index at which to start multiplying (`expression_capacities` is treated as
         * if it started at index `start`.
         */
        __global__ void multiply_capacities(const Util::DeviceArray<EvaluationStatus> statuses,
                                            Util::DeviceArray<size_t> expression_capacities,
                                            const size_t realloc_multiplier,
                                            const size_t expression_count, const size_t start);

        /*
         * @brief Repeats and offsets `expression_capacities` and
         * `expression_capacities_sum` filled with capacities of original array.
         * E.g. original capacities and sums with `[2,4]` and `[2,6]` with `original_expression_count=3` and `original_total_size=7`
         * shall be repeated to `[2,4,1,2,4,1,2,4,...]` and `[2,6,7,9,13,14,16,20,...]` 
         *
         * TODO description
         */
        __global__ void repeat_capacities(Util::DeviceArray<size_t> expression_capacities,
                                          Util::DeviceArray<size_t> expression_capacities_sum,
                                          const size_t original_expression_count,
                                          const size_t original_total_size);

        /*
         * @brief Copies expressions from one array to another while changing offset of each
         * expression
         *
         * @param old_data The array from which the data is copied
         * @param new_data The array to which the data is copied
         * @param old_expression_capacities_sum capacities cumulative sum of `old_data`
         * @param new_expression_capacities_sum capacities cumulative sum of `new_data`
         * @param expression_count Expression count of `old_data`
         */
        __global__ void reoffset_data(const Util::DeviceArray<Symbol> old_data,
                                      Util::DeviceArray<Symbol> new_data,
                                      const Util::DeviceArray<size_t> old_expression_capacities_sum,
                                      const Util::DeviceArray<size_t> new_expression_capacities_sum,
                                      const size_t expression_count);

        /*
         * @brief Sets capacities in `capacities` to the same ones as `other_capacities`, but as if
         * `other_capacities` contained only elements at indices where `scan` changes
         *
         * @param other_capacities Array from which capacities are copied
         * @param scan Inclusive scan of a mask having `1`s at indices at which `other_capacities`
         * elements are considered
         * @param capacities Destination of the copied capacities
         * @param other_size Number of expressions in `ExpressionArray` to which `other_capacities`
         * belongs
         */
        __global__ void reoffset_scan(const Util::DeviceArray<size_t> other_capacities,
                                      const Util::DeviceArray<uint32_t> scan,
                                      Util::DeviceArray<size_t> capacities,
                                      const size_t other_size);

        /*
         * @brief Multiplies capacities and its sum by a constant
         *
         * @param capacities Capacities to multiply
         * @param capacities_sum Capacities sum to multiply
         * @param multiplier The constant the capacities are multiplied by
         */
        __global__ void multiply_capacities(Util::DeviceArray<size_t> capacities,
                                            Util::DeviceArray<size_t> capacities_sum,
                                            const size_t multiplier);
    }

    /*
     * @brief Array of expressions of different lengths in CUDA memory. Each expression begins
     * with a symbol of type T.
     */
    template <class T = Symbol> class ExpressionArray {
        template <class U> friend class ExpressionArray;

        static constexpr size_t KERNEL_BLOCK_SIZE = 512;
        static constexpr size_t KERNEL_BLOCK_COUNT = 8;

        // When doing a reallocation, how many times more memory to allocate than is actually needed
        static constexpr size_t REALLOC_MULTIPLIER = 4;

        Util::DeviceArray<Symbol> data;

        // Swap memory used for reoffsetting. Contains garbage, but is kept at the same size as
        // `data`.
        mutable Util::DeviceArray<Symbol> data_swap;

        // Capacity of every expression. Can potentially have bigger size than there are
        // expressions. In that case, values starting with `expression_count`-th coordinate should
        // be treated as garbage.
        Util::DeviceArray<size_t> capacities;

        // Cumulative sum of `expression_capacities`. Is always of the same size as
        // `expression_capacities`
        Util::DeviceArray<size_t> capacities_sum;

        // Swap memory used for reoffsetting. Contains garbage, but is kept at the same size as
        // `expression_capacities`
        mutable Util::DeviceArray<size_t> capacities_sum_swap;

        // Number of expressions actually held in the array
        size_t expression_count = 0;

        void resize_data(const size_t size) {
            data.resize(size);
            data_swap.resize(size);
        }

        void resize_capacities(const size_t size) {
            capacities.resize(size);
            capacities_sum.resize(size);
            capacities_sum_swap.resize(size);
        }

        __host__ __device__ void expression_capacity_check(const size_t idx) const {
            if constexpr (Consts::DEBUG) {
                if (idx >= expression_count) {
                    Util::crash(
                        "Trying to access capacity of expression %lu of an ExpressionArray with "
                        "size of %lu",
                        idx, expression_count);
                }
            }
        }

        __host__ __device__ void at_check(const size_t idx) const {
            if constexpr (Consts::DEBUG) {
                if (idx >= expression_count) {
                    Util::crash("Trying to access expression %lu of an ExpressionArray with "
                                "size of %lu",
                                idx, expression_count);
                }
            }
        }

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
            data_swap(symbols_capacity),
            capacities(expressions_capacity),
            capacities_sum(expressions_capacity),
            capacities_sum_swap(expressions_capacity) {}

        template <class U>
        ExpressionArray(const ExpressionArray<U>& other) // NOLINT(google-explicit-constructor)
            :
            data(other.data),
            data_swap(other.data_swap),
            capacities(other.capacities),
            capacities_sum(other.capacities_sum),
            capacities_sum_swap(other.capacities_sum_swap),
            expression_count(other.expression_count) {}

        template <class U>
        ExpressionArray(ExpressionArray<U>&& other) // NOLINT(google-explicit-constructor)
            :
            data(std::forward(other.data)),
            data_swap(other.data_swap),
            capacities(std::forward(other.capacities)),
            capacities_sum(std::forward(other.capacities_sum)),
            capacities_sum_swap(other.capacities_sum_swap),
            expression_count(other.expression_count) {}

        template <class U> ExpressionArray& operator=(const ExpressionArray<U>& other) {
            if (&other == this) {
                return *this;
            }

            data = other.data;
            data_swap.resize(other.data_swap);
            capacities = other.capacities;
            capacities_sum = other.capacities_sum;
            capacities_sum_swap.resize(other.capacities_sum_swap);
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
            for (const auto& expression : expressions) {
                current_size_sum += expression.size();
                expressions_sizes.push_back(expression.size());
                expressions_sizes_sum.push_back(current_size_sum);
            }

            resize_capacities(expressions.size());

            cudaMemcpy(capacities.data(), expressions_sizes.data(),
                       expressions_sizes.size() * sizeof(size_t), cudaMemcpyHostToDevice);
            cudaMemcpy(capacities_sum.data(), expressions_sizes_sum.data(),
                       expressions_sizes_sum.size() * sizeof(size_t), cudaMemcpyHostToDevice);

            resize_data(current_size_sum);

            expression_count = expressions.size();

            for (size_t expr_idx = 0; expr_idx < expressions.size(); ++expr_idx) {
                cudaMemcpy(at(expr_idx), expressions[expr_idx].data(),
                           expressions[expr_idx].size() * sizeof(Sym::Symbol),
                           cudaMemcpyHostToDevice);
            }
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
         * @brief Changes the number of held expressions to `new_expression_count`. Current `size()`
         * has to be larger or equal to `new_expression_count`
         */
        void shrink_to(const size_t new_expression_count) {
            if constexpr (Consts::DEBUG) {
                if (new_expression_count > expression_count) {
                    Util::crash(
                        "Trying to shrink an array of %lu elements to a larger size of %lu!",
                        new_expression_count, expression_count);
                }
            }
            expression_count = new_expression_count;
        }

        /*
         * @brief Changes the number of held expressions to `new_expression_count`.
         *
         * @param new_expression_count Number of expressions in the array after the call
         * @param new_expressions_capacity If new expressions will be created, they will all have a
         * capacity of `new_expressions_capacity`
         */
        void resize(const size_t new_expression_count, const size_t new_expressions_capacity) {
            if (new_expression_count <= expression_count) {
                shrink_to(new_expression_count);
                return;
            }

            const size_t additional_expr_count = new_expression_count - expression_count;

            if (capacities.size() < new_expression_count) {
                resize_capacities(new_expression_count * REALLOC_MULTIPLIER);
            }

            ExpressionArrayKernel::
                set_new_expression_capacities<<<KERNEL_BLOCK_COUNT, KERNEL_BLOCK_SIZE>>>(
                    capacities, capacities_sum, expression_count, new_expression_count,
                    new_expressions_capacity);

            const size_t current_total_capacity =
                expression_count == 0 ? 0 : capacities_sum.to_cpu(expression_count - 1);

            const size_t new_required_capacity =
                current_total_capacity + additional_expr_count * new_expressions_capacity;

            if (new_required_capacity > data.size()) {
                resize_data(new_required_capacity * REALLOC_MULTIPLIER);
            }

            expression_count = new_expression_count;
        }

        /*
         * @brief Multiplies capacities of expressions at selected indices by REALLOC_MULTIPLIER.
         * Current content of the array is correctly moved according to new offset values.
         *
         * @param statuses Array of `EvaluationStatus`s. If `indices[i] ==
         * EvaluationStatus::ReallocationRequest`, then the capacity of expression at the
         * `(start+i)`-th index will be multiplied by REALLOC_MULTIPLIER
         * @param start Index at which the reallocations can potentially start
         */
        void reoffset_indices(const Util::DeviceArray<EvaluationStatus> statuses,
                              const size_t start = 0) {
            ExpressionArrayKernel::multiply_capacities<<<KERNEL_BLOCK_COUNT, KERNEL_BLOCK_SIZE>>>(
                statuses, capacities, REALLOC_MULTIPLIER, expression_count, start);
            cudaDeviceSynchronize();

            thrust::inclusive_scan(thrust::device, capacities.begin(), capacities.end(),
                                   capacities_sum_swap.data());
            cudaDeviceSynchronize();

            const size_t new_min_capacity = capacities_sum_swap.to_cpu(expression_count - 1);
            if (new_min_capacity > data.size()) {
                resize_data(new_min_capacity * REALLOC_MULTIPLIER);
            }

            ExpressionArrayKernel::reoffset_data<<<KERNEL_BLOCK_COUNT, KERNEL_BLOCK_SIZE>>>(
                data, data_swap, capacities_sum, capacities_sum_swap, expression_count);
            cudaDeviceSynchronize();

            std::swap(data, data_swap);
            std::swap(capacities_sum, capacities_sum_swap);
        }

        /*
         * @brief Sets expression count and their offsets to the same as the ones in `other`,
         * concatenated `count` times. Current contents of the array turns into garbage.
         *
         * @param other Iterator to an array from which sizes and offsets are going to be copied.
         * Everything will be copied as if the array begun at the element the iterator is pointing
         * to.
         * @param count Number of copies of `other`-like spaces created
         * @param multiplier The capacities copied from `other` are all going to be multiplied by
         * `multiplier`
         */
        template <class U = Symbol>
        void reoffset_like(const typename ExpressionArray<U>::Iterator& other,
                           const size_t multiplier = 1, const size_t count = 1) {
            if (other.expression_count() == 0 || count == 0) {
                expression_count = 0;
                return;
            }

            // Not including the element pointed to
            const size_t other_total_size_up_to_iterator =
                other.index_ == 0 ? 0 : other.array->capacities_sum.to_cpu(other.index_ - 1);

            const size_t other_total_size =
                other.array->capacities_sum.to_cpu(other.array->expression_count - 1);

            // Including the element pointed to
            const size_t other_total_size_past_iterator =
                other_total_size - other_total_size_up_to_iterator;

            if (count * multiplier * other_total_size_past_iterator > data.size()) {
                resize_data(count * other_total_size_past_iterator * REALLOC_MULTIPLIER);
            }

            if (count * other.expression_count() > capacities.size()) {
                resize_capacities(count * other.expression_count() * REALLOC_MULTIPLIER);
            }

            cudaMemcpy(capacities.data(), other.array->capacities.data() + other.index_,
                       other.expression_count() * sizeof(size_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(capacities_sum.data(), other.array->capacities_sum.data() + other.index_,
                       other.expression_count() * sizeof(size_t), cudaMemcpyDeviceToDevice);

            if (count > 1) {
                ExpressionArrayKernel::repeat_capacities<<<KERNEL_BLOCK_COUNT, KERNEL_BLOCK_SIZE>>>(
                    capacities, capacities_sum, other.expression_count(),
                    other_total_size_past_iterator);
            }

            if (multiplier != 1) {
                ExpressionArrayKernel::
                    multiply_capacities<<<KERNEL_BLOCK_COUNT, KERNEL_BLOCK_SIZE>>>(
                        capacities, capacities_sum, multiplier);
            }

            expression_count = count * other.expression_count();
        }

        /*
         * @brief Sets expression count and their offsets to the same as the ones in masked `other`
         * Current contents of the array turns into garbage.
         *
         * @param other Array from which sizes and offsets are going to be copied.
         * @param scan Inclusive scan of a mask with `1`s at the indices that are considered. The
         * array `other` is treated as if it contained only elements at indices `i` where `scan[i]
         * != scan[i - 1]` (or `scan[i] != 0` if `i == 0`).
         */
        template <class U = Symbol>
        void reoffset_like_scan(const ExpressionArray<U>& other,
                                const Util::DeviceArray<uint32_t>& scan) {
            if (other.size() == 0) {
                return;
            }

            const size_t new_expression_count = scan.to_cpu(other.size() - 1);

            if (new_expression_count == 0) {
                expression_count = 0;
                return;
            }

            if (capacities.size() < new_expression_count) {
                resize_capacities(new_expression_count * REALLOC_MULTIPLIER);
            }

            ExpressionArrayKernel::reoffset_scan<<<KERNEL_BLOCK_COUNT, KERNEL_BLOCK_SIZE>>>(
                other.capacities, scan, capacities, other.size());
            cudaDeviceSynchronize();

            thrust::inclusive_scan(thrust::device, capacities.begin(), capacities.end(),
                                   capacities_sum.data());
            cudaDeviceSynchronize();

            const size_t new_total_size = capacities_sum.to_cpu(new_expression_count - 1);

            if (new_total_size > data.size()) {
                resize_data(new_total_size * REALLOC_MULTIPLIER);
            }

            expression_count = new_expression_count;
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

#ifdef __CUDA_ARCH__
        /*
         * @brief Number of `Symbol`s that the `idx`-th expression can potentially hold without
         * changing offsets
         */
        [[nodiscard]] __device__ size_t expression_capacity(const size_t idx) const {
            expression_capacity_check(idx);
            return capacities[idx];
        }
#else
        /*
         * @brief Number of `Symbol`s that the `idx`-th expression can potentially hold without
         * changing offsets
         */
        [[nodiscard]] size_t expression_capacity(const size_t idx) const {
            expression_capacity_check(idx);
            return capacities.to_cpu(idx);
        }
#endif

#ifdef __CUDA_ARCH__
        /*
         * @brief Const pointer to the `idx`-th expression
         */
        __device__ const T* at(const size_t idx) const {
            at_check(idx);

            if (idx == 0) {
                return &data.at(0)->as<T>();
            }

            return &data.at(capacities_sum[idx - 1])->as<T>();
        }
#else
        /*
         * @brief Const pointer to the `idx`-th expression
         */
        const T* at(const size_t idx) const {
            at_check(idx);

            if (idx == 0) {
                return reinterpret_cast<const T*>(data.at(0));
            }

            return reinterpret_cast<const T*>(data.at(capacities_sum.to_cpu(idx - 1)));
        }
#endif

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
            template <class V> friend class ExpressionArray;

            U* array;
            size_t index_;

            __host__ __device__ GenericIterator() : array(nullptr), index_(0) {}

          public:
            __host__ __device__ GenericIterator(U& array, const size_t index) :
                array(&array), index_(index) {
                if constexpr (Consts::DEBUG) {
                    if (index >= array.size()) {
                        Util::crash("Trying to create an iterator to an ExpressionArray past the "
                                    "last expression (index %lu with size %lu)", index, array.size());
                    }
                }
            }

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
            [[nodiscard]] __host__ __device__ T& operator[](const size_t index) const {
                return *(*this)->at(index);
            }

            /*
             * @brief How many expressions onwards there are in the array starting from the one
             * pointed to by the iterator
             */
            [[nodiscard]] __host__ __device__ size_t expression_count() const {
                return array->expression_count - index_;
            }
        };

        using Iterator = GenericIterator<ExpressionArray>;
        using ConstIterator = GenericIterator<const ExpressionArray>;

        /*
         * @brief Constructs an iterator pointing to the index-th expression
         */
        [[nodiscard]] __host__ __device__ Iterator iterator(const size_t index = 0) {
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
