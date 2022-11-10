#ifndef POLYNOMIAL_CUH
#define POLYNOMIAL_CUH

#include "Macros.cuh"

namespace Sym {
    DECLARE_SYMBOL(Polynomial, true)
    size_t rank;

    __host__ __device__ static Polynomial with_rank(size_t rank);

    __host__ __device__ static void divide_polynomials(Polynomial& numerator, Polynomial& denominator, Polynomial& result);

    __host__ __device__ void expand_to(Symbol* destination) const;
    __host__ __device__ inline size_t expanded_size() const;
    __host__ __device__ static size_t expanded_size_from_rank(size_t rank);

    __host__ __device__ bool is_zero() const { return rank == 0 && coefficients()[0] < Util::EPS; }

    __host__ __device__ void make_proper();

    std::string to_string() const;
    std::string to_tex() const;

    DEFINE_IS_POLYNOMIAL(rank)
    DEFINE_IS_MONOMIAL(rank == 0 ? coefficients()[0] : NAN)

    __host__ __device__ double* coefficients();
    __host__ __device__ const double* coefficients() const;
    template <class T>
    __host__ __device__ inline double& operator[](T idx) { return coefficients()[idx]; }
    template <class T>
    __host__ __device__ inline const double& operator[](T idx) const { return coefficients()[idx]; }

    END_DECLARE_SYMBOL(Polynomial)
}

#endif