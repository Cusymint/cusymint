#ifndef ADDITION_CUH
#define ADDITION_CUH

#include <cmath>
#include <vector>

#include "Macros.cuh"

namespace Sym {
    DECLARE_SYMBOL(Addition, false)
    TWO_ARGUMENT_COMMUTATIVE_OP_SYMBOL(Addition)
    std::string to_string() const;
    std::string to_tex() const;

    __host__ __device__ ssize_t is_polynomial(const ssize_t* const ranks) const;
    DEFINE_IS_NOT_MONOMIAL

    /*
     * @brief Inserts in-place `Polynomial` symbol from a sum (which is assumed to be a polynomial).
     * Assumes that structure of Addition is simplified.
     *
     * @param rank Rank of a polynomial calculated with function `is_polynomial()` (must be non-negative).
     * @param help_space A help space for the function (e.g. for creating `Polynomial` symbol).
     */
    //__host__ __device__ void make_polynomial_in_place(int rank, Symbol* const help_space);
    __host__ __device__ void make_polynomial_to(Symbol* const destination, size_t rank);

  private:
    /*
     * @brief Sprawdza, czy `expr1 == sin^2(x)` i `expr2 == cos^2(x)`
     *
     * @return `true` jeśli powyższe to prawda, `false` w.p.p.
     */
    __host__ __device__ static bool is_sine_cosine_squared_sum(const Symbol* const expr1,
                                                               const Symbol* const expr2);

    /*
     * @brief Sprawdza, czy `expr1` i `expr2` są tym samym wyrażeniem, ale o przeciwnym znaku.
     *
     * @param expr1 Pierwsze wyrażenie
     * @param expr2 Drugie wyrażenie
     *
     * @return `true` jeśli `expr1 == -expr2`, `false` w przeciwnym wypadku
     */
    __host__ __device__ static bool are_equal_of_opposite_sign(const Symbol* const expr1,
                                                               const Symbol* const expr2);

    /*
     * @brief W drzewie dodawań usuwa dodawania, których jednym z argumentów jest 0.0. Dodawanie
     * takie jest zamieniane na niezerowy argument lub 0.0, jeśli oba argumenty były zerem.
     * Pozostawia drzewo z niekoherentnymi rozmiarami (wymaga wywołania compress_reverse_to).
     * UWAGA: funkcja ta może zmienić typ `this`, np. kiedy `this == + -1.0 1.0`, to po wykonaniu
     * funkcji `this == 0.0`, czyli typ się zmienił z Addition na NumericConstant! Nie należy więc
     * po wywołaniu tej funkcji wywoływać już żadnych innych funkcji składowych z Addition!
     */
    __host__ __device__ void eliminate_zeros();

    END_DECLARE_SYMBOL(Addition)

    DECLARE_SYMBOL(Negation, false)
    ONE_ARGUMENT_OP_SYMBOL
    std::string to_string() const;
    std::string to_tex() const;
    __host__ __device__ ssize_t is_polynomial(const ssize_t* const ranks) const;
    __host__ __device__ double get_monomial_coefficient(const double* const coefficients) const;
    END_DECLARE_SYMBOL(Negation)

    std::vector<Symbol> operator+(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs);
    std::vector<Symbol> operator-(const std::vector<Symbol>& arg);
    std::vector<Symbol> operator-(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs);
}

#endif
