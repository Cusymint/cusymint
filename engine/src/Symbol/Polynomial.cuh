#ifndef POLYNOMIAL_CUH
#define POLYNOMIAL_CUH

#include "Macros.cuh"

namespace Sym {
    DECLARE_SYMBOL(Polynomial, true)
    int rank;

    __host__ __device__ static Polynomial with_rank(int rank);

    __host__ __device__ static void divide_polynomials(Polynomial numerator, Polynomial denominator, Polynomial result);

    std::string to_string() const;
    std::string to_tex() const;

    DEFINE_IS_POLYNOMIAL(rank)
    DEFINE_IS_MONOMIAL(rank == 0 ? coefficients()[0] : NAN)

    __host__ __device__ double* coefficients();
    __host__ __device__ const double* coefficients() const;
    __host__ __device__ inline double& operator[](int idx) { return coefficients()[idx]; }
    __host__ __device__ inline const double& operator[](int idx) const { return coefficients()[idx]; }

    END_DECLARE_SYMBOL(Polynomial)
}

#endif