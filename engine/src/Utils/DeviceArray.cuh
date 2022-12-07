#ifndef DEVICE_ARRAY_CUH
#define DEVICE_ARRAY_CUH

#include <type_traits>
#include <vector>

#include "CompileConstants.cuh"
#include "Cuda.cuh"

namespace Util {
    template <class T> class DeviceArray;

    static constexpr size_t SET_MEM_BLOCK_SIZE = 1024;

    /*
     * @brief Sets all elements of `array` to `val` in parallel
     */
    template <class T> __global__ void set_mem_kernel(DeviceArray<T> array, const T val) {
        const size_t idx = Util::thread_idx();

        if (idx < array.size()) {
            array[idx] = val;
        }
    }

    /*
     * @brief A class facilitating the use of CUDA memory.
     * The memory is owned by the array allocating it. All copies of the array only copy the pointer
     * and cannot deallocate or reallocate the memory. This allows passing DeviceArrays directly as
     * arguments to CUDA kernels.
     */
    template <class T> class DeviceArray {
        // Number of elements of type T in the array
        size_t data_size = 0;
        T* data_ptr = nullptr;
        bool is_data_owner = false;

        void allocate_memory() {
            cudaError_t result = cudaMalloc(&data_ptr, size_in_bytes());
            if (result == cudaErrorMemoryAllocation) {
                throw std::bad_alloc();
            }
        }

        void free_data() {
            if (is_data_owner) {
                cudaFree(data_ptr);
            }
        }

      public:
        /*
         * @brief Creates an empty `DeviceArray`
         */
        DeviceArray() : is_data_owner(true) {}

        /*
         * @brief Allocates an array of size `size`
         *
         * @param zero_mem If `true`, the allocated memory is zeroed
         */
        explicit DeviceArray(const size_t size, const bool zero_mem = false) :
            data_size(size), is_data_owner(true) {
            allocate_memory();

            if (zero_mem) {
                this->zero_mem();
            }
        }

        /*
         * @brief Creates a DeviceArray from a vector
         *
         * @param vector A vector from which size and data are copied
         */
        explicit DeviceArray(const std::vector<T>& vector) :
            data_size(vector.size()), is_data_owner(true) {
            allocate_memory();
            cudaMemcpy(data_ptr, vector.data(), size_in_bytes(), cudaMemcpyHostToDevice);
        }

        DeviceArray(const DeviceArray& other) :
            data_size(other.data_size), data_ptr(other.data_ptr) {}

        DeviceArray(DeviceArray&& other) noexcept :
            data_size(other.data_size),
            data_ptr(other.data_ptr),
            is_data_owner(other.is_data_owner) {
            other.data_ptr = nullptr;
            other.data_size = 0;
            other.is_data_owner = false;
        }

        DeviceArray& operator=(const DeviceArray& other) {
            if (&other == this) {
                return *this;
            }

            free_data();

            data_size = other.data_size;
            data_ptr = other.data_ptr;
            is_data_owner = false;

            return *this;
        }

        DeviceArray& operator=(DeviceArray&& other) noexcept {
            free_data();

            data_size = other.data_size;
            data_ptr = other.data_ptr;
            is_data_owner = other.is_data_owner;

            other.data_size = 0;
            other.data_ptr = nullptr;
            other.is_data_owner = false;

            return *this;
        }

        /*
         * @brief Reallocates the memory. The array has to own the memory it points to. If the new
         * size is greater or equal to the old size, all elements are copied over. If not, then only
         * the ones that fit (starting with the first one) are copied.
         *
         * @param new_size New size of the array (number of elements of type T)
         */
        void resize(const size_t new_size) {
            if (!is_data_owner) {
                throw std::logic_error("Resizing an unowned array");
            }

            // Everything works also when `data_ptr == nullptr`
            T* const old_data_ptr = data_ptr;
            const size_t old_size = data_size;

            data_size = new_size;
            allocate_memory();

            cudaMemcpy(data_ptr, old_data_ptr, std::min(new_size, old_size) * sizeof(T),
                       cudaMemcpyDeviceToDevice);

            cudaFree(old_data_ptr);
        }

        /*
         * @brief Creates std::vector with contets of the array
         */
        std::vector<T> to_vector() const {
            std::vector<T> vector(data_size);
            cudaMemcpy(vector.data(), data_ptr, size_in_bytes(), cudaMemcpyDeviceToHost);
            return vector;
        }

        /*
         * @brief Copies one value from the array to the cpu
         *
         * @param idx Index from which to copy the value
         *
         * @param copied value
         */
        T to_cpu(const size_t idx) const {
            T value;
            cudaMemcpy(&value, at(idx), sizeof(T), cudaMemcpyDeviceToHost);
            return value;
        }

        /*
         * @brief Zeros the array
         */
        inline void zero_mem() { cudaMemset(data_ptr, 0, size_in_bytes()); }

        /*
         * @brief Sets value of all elements without synchronizing the device
         *
         * @param val Value to assign to each element
         */
        void set_mem_async(const T& val) {
            if (size() == 0) {
                return;
            }

            const size_t block_count = (size() - 1) / (SET_MEM_BLOCK_SIZE) + 1;
            set_mem_kernel<<<block_count, SET_MEM_BLOCK_SIZE>>>(*this, val);
        }

        /*
         * @brief Sets value of all elements and synchronizes the device
         *
         * @param val Value to assign to each element
         */
        void set_mem(const T& val) {
            set_mem_async(val);
            cudaDeviceSynchronize();
        }

        /*
         * @brief Array size in bytes
         */
        [[nodiscard]] __host__ __device__ inline size_t size_in_bytes() const {
            return sizeof(T) * data_size;
        }

        /*
         * @brief Array size
         */
        [[nodiscard]] __host__ __device__ inline size_t size() const { return data_size; }

        /*
         * @brief Array memory pointer
         */
        __host__ __device__ inline const T* data() const { return data_ptr; }

        /*
         * @brief Array memory pointer
         */
        __host__ __device__ inline T* data() {
            return const_cast<T*>(const_cast<const DeviceArray<T>*>(this)->data());
        }

        /*
         * @brief Const pointer to the element right behind the array
         */
        __host__ __device__ inline const T* cend() const { return data_ptr + data_size; }

        /*
         * @brief Pointer to the element right behind the array
         */
        __host__ __device__ inline T* end() { return const_cast<T*>(cend()); }

        /*
         * @brief Const pointer to the first element of the array
         */
        __host__ __device__ inline const T* cbegin() const { return data(); }

        /*
         * @brief Pointer to the first element of the array
         */
        __host__ __device__ inline T* begin() { return const_cast<T*>(cbegin()); }

        /*
         * @brief Const pointer to the `idx`-th element of the array
         */
        __host__ __device__ inline const T* at(const size_t idx) const {
            if constexpr (Consts::DEBUG) {
                if (idx > data_size) {
                    crash("Trying to access element %lu of an array of %lu elements", idx,
                          data_size);
                }

                if (data_ptr == nullptr) {
                    crash("Trying to index an array that does not point to any data.");
                }
            }

            return data_ptr + idx;
        }

        /*
         * @brief Pointer to the `idx`-th element of the array
         */
        __host__ __device__ inline T* at(const size_t idx) {
            return const_cast<T*>(const_cast<const DeviceArray<T>*>(this)->at(idx));
        }

        /*
         * @brief Const reference to the `idx`-th element of the array
         */
        __device__ inline const T& operator[](const size_t idx) const { return *at(idx); }

        /*
         * @brief Reference to the `idx`-th element of the array
         */
        __device__ inline T& operator[](const size_t idx) {
            return const_cast<T&>(const_cast<const DeviceArray<T>&>(*this)[idx]);
        }

        /*
         * @brief Pointer to the last element of the array
         */
        inline T* last() { return at(data_size - 1); }

        /*
         * @brief Value of the last element of the array moved to the CPU
         */
        inline T last_cpu() const { return to_cpu(data_size - 1); }

        ~DeviceArray() { free_data(); }
    };
}

#endif
