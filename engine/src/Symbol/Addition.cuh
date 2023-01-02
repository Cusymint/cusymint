#ifndef ADDITION_CUH
#define ADDITION_CUH

#include <vector>

#include "Macros.cuh"

namespace Sym {
    DECLARE_SYMBOL(Addition, false)
    TWO_ARGUMENT_COMMUTATIVE_OP_SYMBOL(Addition)
    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::string to_tex() const;

    /*
     * @brief If the `symbol` is in the form `a*f`, where `a` is a `NumericConstant` and `a` is an
     * expression that is NOT a `Product`, then returns reference to `f` and sets `coefficient` to
     * `a`
     */
    __host__ __device__ static const Sym::Symbol&
    extract_base_and_coefficient(const Sym::Symbol& symbol, double& coefficient);

    /*
     * @brief If the expression can be expressed as `a*f(x)`, where `a` is `NumericConstant`,
     * returns the value of `a`
     * @param `symbol` given expression
     *
     * @return Value of `a` or `1` if `symbol` can't be expressed as `a*f(x)`.
     */
    __host__ __device__ static double coefficient(const Sym::Symbol& symbol);

    /*
     * @brief Copies a Product tree skipping any `NumericConstant`s
     *
     * @param dst Destination
     * @param expr Expression to copy
     */
    __host__ __device__ static void copy_without_coefficient(Sym::Symbol& dst,
                                                             const Sym::Symbol& expr);

    /*
     * @brief Copies a Product with a new coefficient, skipping any existing one
     *
     * @param dst Destination
     * @param expr Expression to copy
     * @param coeff New coefficient
     */
    __host__ __device__ static void copy_with_coefficient(Sym::Symbol& dst, const Sym::Symbol& expr,
                                                          const double coeff);

    /*
     * @brief Checks if `f == g`, where `f*a == expr1` and `g*b == expr2`, where `a` and `b` are
     * `NumericConstant`s
     *
     * @param expr1 First expression
     * @param expr2 Second expression
     *
     * @return `true` if `f == g`, `false` otherwise
     */
    __host__ __device__ static bool are_equal_except_for_constant(const Sym::Symbol& expr1,
                                                                  const Sym::Symbol& expr2);

    /*
     * @brief Checks the ordering of `f` and `g`, where `f*a == expr1` and `g*b == expr2`, where `a`
     * and `b` are `NumericConstant`s
     *
     * @param expr1 First expression
     * @param expr2 Second expression
     * @param help_space Help space
     *
     * @return Ordering of `f` and `g`
     */
    __host__ __device__ static Util::Order compare_except_for_constant(const Sym::Symbol& expr1,
                                                                       const Sym::Symbol& expr2,
                                                                       Symbol& help_space);

    /*
     * @brief Checks if `expr1` and `expr2` are the same expression but with an opposite sign
     *
     * @param expr1 First expression
     * @param expr2 Second expression
     *
     * @return `true` if `expr1 == -expr2`, `false` otherwise
     */
    __host__ __device__ static bool are_equal_of_opposite_sign(const Symbol& expr1,
                                                               const Symbol& expr2);

  private:
    /*
     * @brief Checks if `expr1 == sin^2(x)` and `expr2 == cos^2(x)`.
     *
     * @return `true` if aforementioned is true, `false` otherwise.
     */
    __host__ __device__ static bool is_sine_cosine_squared_sum(const Symbol* const expr1,
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

    std::vector<Symbol> operator+(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs);
    std::vector<Symbol> operator-(const std::vector<Symbol>& arg);
    std::vector<Symbol> operator-(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs);
}

#endif
