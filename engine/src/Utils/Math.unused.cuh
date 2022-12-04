#ifndef MATH_CUH
#define MATH_CUH

#include "Utils/CompileConstants.cuh"
#include "Utils/Cuda.cuh"
#include <cstddef>
#include <cuda/std/complex>
#include <type_traits>

namespace Util::Math {

    __host__ __device__ bool is_integer(double d) { return floor(d) == d; }
    __host__ __device__ bool is_squared_integer(double d) { return is_integer(sqrt(d)); }
    __host__ __device__ bool is_sum_of_squares(int num, int& out1, int& out2) {
        for (int i = 0; i < num / 2; ++i) {
            const double root = sqrt(num - i * i);
            if (is_integer(root)) {
                out1 = i;
                out2 = static_cast<int>(root);
                return true;
            }
        }
        return false;
    }

    template <class T = int> //, std::enable_if_t<std::is_integral_v<T>, void>* = nullptr>
    __host__ __device__ T gcd(T num1, T num2) {
        T rest;
        while (num2 > 0) {
            rest = num1 % num2;
            num1 = num2;
            num2 = rest;
        }
        return num1;
    }

    template <class T = int> //, std::enable_if_t<std::is_integral_v<T>, void>* = nullptr>
    class ComplexRational {
        T numerator_real;
        T numerator_imag;
        T denominator;

      public:
        __host__ __device__ ComplexRational(const T& numerator_real, const T& numerator_imag,
                                            const T& denominator) :
            numerator_real(numerator_real),
            numerator_imag(numerator_imag),
            denominator(denominator) {}

        __host__ __device__ void reduce_by_gcd() {
            const T divisor = gcd(denominator, gcd(numerator_real, numerator_imag));
            numerator_real /= divisor;
            numerator_imag /= divisor;
            denominator /= divisor;
        }

        __host__ __device__ cuda::std::complex<double> get_complex() {
            return {static_cast<double>(numerator_real) / denominator,
                    static_cast<double>(numerator_imag) / denominator};
        };
    };

    template <class T = int> //, std::enable_if_t<std::is_integral_v<T>, void>* = nullptr>
    class DivisorIterator {
        T number;
        T current_divisor;

      public:
        explicit DivisorIterator(const T& number) : number(number), current_divisor(1) {}

        __host__ __device__ bool is_valid() const { return current_divisor > number; }

        __host__ __device__ T current() const {
            if (Consts::DEBUG) {
                if (!is_valid()) {
                    Util::crash(
                        "Trying to access the current element of an exhausted divisor iterator");
                }
            }
            return current_divisor;
        }

        __host__ __device__ bool advance() {
            while (current_divisor <= number) {
                if (number % current_divisor == 0) {
                    return true;
                }
                current_divisor++;
            }
            return false;
        }
    };

    struct ComplexPolynomial {
        size_t rank;
        cuda::std::complex<double>* data;
    };

    /*
     * @brief Checks if `root_candidate` is a root of `polynomial`, using synthetic division scheme.
     * If yes, writes result to `polynomial` and returns `true`. Otherwise returns `false`.
     *
     * @param `polynomial` Input polynomial
     * @param `root_candidate` Number to be checked 
     * @param `help_space` Help space for calculations. There must be enough memory for `polynomial.rank`
     * numbers.
     *
     * @return `true` if operation succeeds, `false` otherwise.
     */
    __host__ __device__ bool try_find_root(ComplexPolynomial& polynomial,
                                           const cuda::std::complex<double>& root_candidate,
                                           cuda::std::complex<double>* const help_space) {
        if (polynomial.rank == 0) {
            return false;
        }
        help_space[0] = polynomial.data[0];
        for (size_t i = 1; i <= polynomial.rank; ++i) {
            help_space[i] = help_space[i - 1] * root_candidate + polynomial.data[i];
        }
        if (cuda::std::abs(help_space[polynomial.rank]) < Consts::EPS) {
            Util::copy_mem(polynomial.data, help_space, polynomial.rank * sizeof(cuda::std::complex<double>));
            return true;
        }
        return false;
    }
}

#endif