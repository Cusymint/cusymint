#ifndef DEVICE_ARRAY_CUH
#define DEVICE_ARRAY_CUH

#include <type_traits>

#include "Cuda.cuh"

namespace Util {
    /*
     * @brief Klasa ułatwiająca korzystanie z pamięci CUDA.
     * Pozwala na używanie tylko T które są POD (kopie wykonywane są kopiowaniem surowej pamięci).
     */
    template <class T> class DeviceArray {
        static_assert(
            std::is_pod<T>::value,
            "Type contained in DeviceArray has to be POD, but a different type has been provided.");

        size_t data_size = 0;
        T* data = nullptr;

        void allocate_data() { cudaMalloc(&data, size_in_bytes()); }

      public:
        /*
         * @brief Alokuje tablicę rozmiaru `size`
         */
        explicit DeviceArray(const size_t size) : data_size(size) { allocate_data(); }

        DeviceArray(const DeviceArray& other) : data_size(other.data_size) {
            allocate_data();
            copy_mem(data, other.data, sizeof(T) * data_size);
        }

        DeviceArray(DeviceArray&& other) noexcept : data_size(other.data_size), data(other.data) {
            other.data = nullptr;
            other.data_size = 0;
        }

        DeviceArray& operator=(const DeviceArray& other) {
            if (&other == this) {
                return *this;
            }

            data_size = other.data_size;
            allocate_data();
            copy_mem(data, other.data, sizeof(T) * data_size);
        }

        DeviceArray& operator=(DeviceArray&& other) noexcept {
            data_size = other.data_size;
            other.data_size = 0;
            data = other.data;
            other.data = nullptr;
        }

        ~DeviceArray() { cudaFree(data); }

        /*
         * @brief Zwraca rozmiar tablicy
         */
        __host__ __device__ size_t size() { return data_size; }

        /*
         * @brief Zwraca rozmiar tablicy w bajtach
         */
        __host__ __device__ size_t size_in_bytes() { return sizeof(T) * data_size; }

        /*
         * @brief Referencję do idx-tego elementu tablicy
         */
        __device__ T& operator[](const size_t idx) { return data[idx]; }

        /*
         * @brief Referencję do idx-tego elementu tablicy
         */
        __device__ const T& operator[](const size_t idx) const { return data[idx]; }
    };
}

#endif
