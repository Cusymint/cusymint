#ifndef PRODUCT_CUH
#define PRODUCT_CUH

#include <vector>

#include "Macros.cuh"
#include "Symbol/SimplificationResult.cuh"

namespace Sym {
    DECLARE_SYMBOL(Product, false)
    TWO_ARGUMENT_COMMUTATIVE_OP_SYMBOL(Product)
    std::string to_string() const;
    std::string to_tex() const;

    /*
     * @brief W drzewie mnożenia usuwa mnożenia, których jednym z argumentów jest 1.0. Mnożenie
     * takie jest zamieniane na niejedynkowy argument lub 1.0, jeśli oba argumenty były jedynką.
     * Pozostawia drzewo z niekoherentnymi rozmiarami (wymaga wywołania compress_reverse_to).
     * UWAGA: funkcja ta może zmienić typ `this`, np. kiedy `this == * Reciprocal(pi) pi`, to po
     * wykonaniu funkcji `this == 1.0`, czyli typ się zmienił z Product na NumericConstant! Nie
     * należy więc po wywołaniu tej funkcji wywoływać już żadnych innych funkcji składowych z
     * Product!
     */
    __host__ __device__ void eliminate_ones();

  private:
    /*
     * @brief Sprawdza, czy `expr1` i `expr2` są swoimi odwrotnościami.
     *
     * @param expr1 Pierwsze wyrażenie
     * @param expr2 Drugie wyrażenie
     *
     * @return `true` jeśli `expr1 == 1/expr2`, `false` w przeciwnym wypadku
     */
    __host__ __device__ static bool are_inverse_of_eachother(const Symbol& expr1,
                                                             const Symbol& expr2);

    /*
     * @brief Checks if `this` is a rational function with numerator rank higher than
     * denominator rank and tries to transform it by dividing numerator by denominator.
     * Does not simplify the fraction by GCD.
     *
     * @param `expr1` First expression
     * @param `expr2` Second expression
     * @param `help_space` a help space
     *
     * @return `NeedsSimplification` if division was successful, `NoAction` if didn't happen,
     * `NeedsSpace` if division requires additional space. Never returns `Success`.
     */
    __host__ __device__ static SimplificationResult
    try_dividing_polynomials(Symbol* const expr1, Symbol* const expr2, Symbol* const help_space);

    /*
     * @brief Checks if one of the factors is `Addition` and splits `this` using the rule
     * `a(b+c) -> ab+ac`.
     *
     * @param `expr1` First expression
     * @param `expr2` Second expression
     * @param `help_space` a help space
     *
     * @return `NeedsSimplification` if transformation was successful, `NoAction` if didn't happen,
     * `NeedsSpace` if transformation requires additional space. Never returns `Success`.
     */
    __host__ __device__ static SimplificationResult
    try_split_into_sum(Symbol* const expr1, Symbol* const expr2, Symbol* const help_space);

    END_DECLARE_SYMBOL(Product)

    DECLARE_SYMBOL(Reciprocal, false)
    ONE_ARGUMENT_OP_SYMBOL

    std::string to_string() const;
    std::string to_tex() const;

    END_DECLARE_SYMBOL(Reciprocal)

    std::vector<Symbol> operator*(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs);
    std::vector<Symbol> inv(const std::vector<Symbol>& arg);
    std::vector<Symbol> operator/(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs);
}

#endif
