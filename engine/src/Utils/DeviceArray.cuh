#ifndef DEVICE_ARRAY_CUH
#define DEVICE_ARRAY_CUH

#include <type_traits>
#include <vector>

#include "Cuda.cuh"

namespace Util {
    /*
     * @brief Klasa ułatwiająca korzystanie z pamięci CUDA.
     * Właścicielem pamięci jest tylko tablica alokująca pamięć. Wszystkie kopie tylko kopiują
     * wskaźnik (rozwiązanie ze względu na przekazywanie elementów tej klasy do kerneli CUDA)
     */
    template <class T> class DeviceArray {
        // Liczba elementów T w tablicy
        size_t data_size = 0;
        T* data_ptr = nullptr;
        bool is_data_owner = false;

        void allocate_data() { cudaMalloc(&data_ptr, size_in_bytes()); }

        void free_data() {
            if (is_data_owner) {
                cudaFree(data_ptr);
            }
        }

      public:
        /*
         * @brief Alokuje tablicę rozmiaru `size`
         *
         * @param zero_mem Jeśli `true`, to zaalokowana pamięć jest zerowana
         */
        explicit DeviceArray(const size_t size, const bool zero_mem = false) :
            data_size(size), is_data_owner(true) {
            allocate_data();

            if (zero_mem) {
                this->zero_mem();
            }
        }

        /*
         * @brief Alokuje tablicę rozmiaru `size` i zeruje pamięć
         */
        DeviceArray(const size_t size, const T& value) : data_size(size), is_data_owner(true) {
            allocate_data();
            for (size_t i = 0; i < size; ++i) {
                data_ptr[i] = value;
            }
        }

        explicit DeviceArray(const std::vector<T>& vector) :
            data_size(vector.size()), is_data_owner(true) {
            allocate_data();
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
         * @brief Tworzy std::vector i kopiuje do niego swoją zawartość
         */
        std::vector<T> to_vector() const {
            std::vector<T> vector(data_size);
            cudaMemcpy(vector.data(), data_ptr, size_in_bytes(), cudaMemcpyDeviceToHost);
            return vector;
        }

        inline void zero_mem() { cudaMemset(data_ptr, 0, size_in_bytes()); }

        /*
         * @brief Rozmiar tablicy w bajtach
         */
        __host__ __device__ inline size_t size_in_bytes() const { return sizeof(T) * data_size; }

        /*
         * @brief Rozmiar tablicy
         */
        __host__ __device__ inline size_t size() const { return data_size; }

        /*
         * @brief Wskaźnik do pamięci tablicy
         */
        __host__ __device__ inline const T* data() const { return data_ptr; }

        /*
         * @brief Wskaźnik do pamięci tablicy
         */
        __host__ __device__ inline T* data() {
            return const_cast<T*>(const_cast<const DeviceArray<T>*>(this)->data());
        }

        /*
         * @brief Stały wskaźnik do elementu o jeden za tablicą
         */
        __host__ __device__ inline const T* cend() const { return data_ptr + data_size; }

        /*
         * @brief Wskaźnik do elementu o jeden za tablicą
         */
        __host__ __device__ inline T* end() { return const_cast<T*>(cend()); }

        /*
         * @brief Stały wskaźnik do pierwszego elementu tablicy
         */
        __host__ __device__ inline const T* cbegin() const { return data(); }

        /*
         * @brief Wskaźnik do pierwszego elementu tablicy
         */
        __host__ __device__ inline T* begin() { return const_cast<T*>(cbegin()); }

        /*
         * @brief Referencja do idx-tego elementu tablicy
         */
        __device__ inline const T& operator[](const size_t idx) const { return data_ptr[idx]; }

        /*
         * @brief Referencja do idx-tego elementu tablicy
         */
        __device__ inline T& operator[](const size_t idx) {
            return const_cast<T&>(const_cast<const DeviceArray<T>&>(*this)[idx]);
        }

        /*
         * @brief Wskaźnik do idx-tego elementu tablicy
         */
        __host__ __device__ inline const T* at(const size_t idx) const { return data_ptr + idx; }

        /*
         * @brief Wskaźnik do idx-tego elementu tablicy
         */
        __host__ __device__ inline T* at(const size_t idx) {
            return const_cast<T*>(const_cast<const DeviceArray<T>*>(this)->at(idx));
        }

        /*
         * @brief Wskaźnik do ostatniego elementu tablicy
         */
        inline T* last() { return at(data_size - 1); }

        ~DeviceArray() { free_data(); }
    };
}

#endif
