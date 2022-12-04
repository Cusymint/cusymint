#ifndef POLYNOMIAL_CUH
#define POLYNOMIAL_CUH

#include "Macros.cuh"
#include "Symbol/Addition.cuh"
#include <cstddef>

namespace Sym {
    DECLARE_SYMBOL(Polynomial, true)
    size_t rank;

    __host__ __device__ static Polynomial with_rank(size_t rank);

    /*
     * @brief Creates `Polynomial` from a `symbol` (which is assumed to be a polynomial).
     * Assumes that structure of `symbol` is simplified.
     *
     * @param symbol Symbol which has been checked with `is_polynomial` function
     * @param destination Pointer to memory where `Polynomial` symbol will be created.
     */
    __host__ __device__ static void make_polynomial_at(const Symbol* const symbol,
                                                       Symbol* const destination);

    /*
     * @brief Divides polynomials `numerator` by `denominator` and modifies `numerator` and `result`
     * according to formula `p(x)/q(x) = s(x) + r(x)/q(x)`.
     * `numerator`'s rank must be greater than `denominator`'s, and rank of `result` must be equal
     * to difference of `numerator`'s and `denominator`'s ranks.
     *
     * @param `numerator` Dividend of the operation (`p(x)`, shall be updated to `r(x)`)
     * @param `denominator` Divisor of the operation (`q(x)`)
     * @param `result` Result of the operation (shall be updated to `s(x)`)
     */
    __host__ __device__ static void divide_polynomials(Polynomial& numerator,
                                                       Polynomial& denominator, Polynomial& result);

    __host__ __device__ void expand_to(Symbol* destination) const;
    [[nodiscard]] __host__ __device__ inline size_t expanded_size() const;
    __host__ __device__ static size_t expanded_size_from_rank(size_t rank);

    [[nodiscard]] __host__ __device__ bool is_zero() const {
        return rank == 0 && coefficients()[0] < Consts::EPS;
    }

    __host__ __device__ void make_proper();

    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::string to_tex() const;

    [[nodiscard]] __host__ __device__ double* coefficients();
    [[nodiscard]] __host__ __device__ const double* coefficients() const;

    template <class T> __host__ __device__ inline double& operator[](T idx) {
        return coefficients()[idx];
    }

    template <class T> __host__ __device__ inline const double& operator[](T idx) const {
        return coefficients()[idx];
    }

    END_DECLARE_SYMBOL(Polynomial)
}

#endif
