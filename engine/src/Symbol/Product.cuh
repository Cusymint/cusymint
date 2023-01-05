#ifndef PRODUCT_CUH
#define PRODUCT_CUH

#include <vector>

#include "Macros.cuh"
#include "Symbol/SimplificationResult.cuh"

namespace Sym {
    DECLARE_SYMBOL(Product, false)
    TWO_ARGUMENT_COMMUTATIVE_OP_SYMBOL(Product)
    std::string to_string() const;
    std::string to_tex_without_negation() const;
    std::string to_tex() const;

    /*
     * @brief W drzewie mnożenia usuwa mnożenia, których jednym z argumentów jest 1.0. Mnożenie
     * takie jest zamieniane na niejedynkowy argument lub 1.0, jeśli oba argumenty były jedynką.
     * Pozostawia drzewo z niekoherentnymi rozmiarami (wymaga wywołania compress_reverse_to).
     * WARNING: this function can change the type of `this`, e.g. if `this == * Reciprocal(pi) pi`,
     * then after the call `this == 1.0`, so the type changed to `NumericConstant`! Don't call any
     * other `Product` member functions after calling this one!
     */
    __host__ __device__ void eliminate_ones();

    /*
     * @brief If the expression can be expressed as `f(x)^(ag(x))`, where `a` is `NumericConstant`,
     * returns the value of `a`
     *
     * @param `symbol` given expression
     *
     * @return Value of `a` or `1` if `symbol` can't be expressed as `f(x)^(ag(x))`.
     */
    __host__ __device__ static double exponent_coefficient(const Symbol& symbol);

    /*
     * @brief If the expression is a power, returns a reference to its base. If not, returns the
     * expression itself.
     *
     * @param `symbol` given expression.
     *
     * @return Reference to the expression or its base
     */
    __host__ __device__ static const Symbol& base(const Symbol& symbol);

    /*
     * @brief If the expression is a power, returns a reference to its exponent. If not, returns a
     * reference to the static identity function
     *
     * @param `symbol` given expression.
     *
     * @return Reference to the expression's exponent or `NumericConstant` with value of `1.0`
     */
    __host__ __device__ static const Symbol& exponent(const Symbol& symbol);

  private:
    /*
     * @brief Checks, if `expr1` and `expr2` are an inverse of eachother
     *
     * @param expr1 First expression
     * @param expr2 Second expression
     *
     * @return `true` if `expr1 == 1/expr2`, `false` otherwise
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

    std::vector<Symbol> operator*(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs);
    std::vector<Symbol> inv(const std::vector<Symbol>& arg);
    std::vector<Symbol> operator/(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs);
}

#endif
