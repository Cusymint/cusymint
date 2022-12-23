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
     * @brief Extracts expression `f(x)` from `symbol`, where `symbol` is like `-f(x)` or `a*f(x)`.
     * Sets a `coefficient` such that `coefficient*f(x) = symbol`.
     *
     * @param `symbol` given expression
     * @param `coefficient` coefficient to be set
     *
     * @return extracted expression from `symbol`
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
     * @brief Sprawdza, czy `expr1 == sin^2(x)` i `expr2 == cos^2(x)`
     *
     * @return `true` jeśli powyższe to prawda, `false` w.p.p.
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
