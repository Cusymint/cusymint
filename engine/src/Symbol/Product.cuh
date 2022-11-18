#ifndef PRODUCT_CUH
#define PRODUCT_CUH

#include <vector>

#include "Macros.cuh"

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
     * @brief Checks if `this` is a rational function and tries to transform it
     * by dividing numerator by denominator. Does not simplify the fraction by GCD.
     *
     * @param `help_space` a help space
     *
     * @return `true` if division was successful or didn't happen.
     * `false` if division requires additional space.
     */
    __host__ __device__ bool try_dividing_polynomials(Symbol* const help_space);

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
