#ifndef DEVICE_ARRAY_CUH
#define DEVICE_ARRAY_CUH

#include <type_traits>

#include "Cuda.cuh"

namespace Util {
    /*
     * @brief Klasa ułatwiająca korzystanie z pamięci CUDA.
     * Właścicielem pamięci jest tylko tablica alokująca pamięć. Wszystkie kopie tylko kopiują
     * wskaźnik (rozwiązanie ze względu na przekazywanie elementów tej klasy do kerneli CUDA)
     */
    template <class T> class DeviceArray {
        size_t data_size = 0;
        T* data = nullptr;
        bool is_data_owner = false;

        void allocate_data() { cudaMalloc(&data, size_in_bytes()); }

      public:
        /*
         * @brief Alokuje tablicę rozmiaru `size`
         */
        explicit DeviceArray(const size_t size) : data_size(size), is_data_owner(true) {
            allocate_data();
        }

        DeviceArray(const DeviceArray& other) : data_size(other.data_size), data(other.data) {}

        DeviceArray(DeviceArray&& other) noexcept :
            data_size(other.data_size), data(other.data), is_data_owner(other.is_data_owner) {
            other.data = nullptr;
            other.data_size = 0;
            other.is_data_owner = false;
        }

        DeviceArray& operator=(const DeviceArray& other) {
            if (&other == this) {
                return *this;
            }

            data_size = other.data_size;
            data = other.data;
            is_data_owner = false;
        }

        DeviceArray& operator=(DeviceArray&& other) noexcept {
            data_size = other.data_size;
            data = other.data;
            is_data_owner = other.is_data_owner;

            other.data_size = 0;
            other.data = nullptr;
            other.is_data_owner = false;
        }

        ~DeviceArray() {
            if (is_data_owner) {
                cudaFree(data);
            }
        }

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
