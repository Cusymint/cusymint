#ifndef SUBSTITUTION_CUH
#define SUBSTITUTION_CUH

#include <vector>

#include "Macros.cuh"

namespace Sym {
    DECLARE_SYMBOL(Substitution, false)
    static const size_t SUBSTITUTION_NAME_COUNT;
    static const char* const SUBSTITUTION_NAMES[];

    size_t substitution_idx;

    __host__ __device__ static void create(const Symbol* const expression,
                                           Symbol* const destination,
                                           const size_t substitution_idx);
    [[nodiscard]] static std::string nth_substitution_name(const size_t n);

    [[nodiscard]] __host__ __device__ const Symbol& expression() const;
    [[nodiscard]] __host__ __device__ Symbol& expression();
    [[nodiscard]] __host__ __device__ const Substitution& next_substitution() const;
    [[nodiscard]] __host__ __device__ Substitution& next_substitution();
    [[nodiscard]] __host__ __device__ bool is_last_substitution() const;

    /*
     * @brief Changes all occurrences of `Variable` in the expression to a name from
     * `SUBSTITUTION_NAMES` based on the value of `substitution_idx
     *
     * @return Expression with the constant substituted
     */
    [[nodiscard]] std::vector<Symbol> expression_with_constant() const;

    /*
     * @brief Substitution as a string, but without subsequent substitutions
     *
     * @return String representing the substitution
     */
    [[nodiscard]] std::string to_string_this() const;
    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::string to_tex_this() const;
    [[nodiscard]] std::string to_tex() const;

    /*
     * @brief Applies this substitution to `expession`
     *
     * @param expression Expression where variable will be replaced by contents of `this`
     *
     * @return `expression` after substitution
     */
    [[nodiscard]] std::vector<Symbol> substitute(std::vector<Symbol> expr) const;
    END_DECLARE_SYMBOL(Substitution)
}

#endif
