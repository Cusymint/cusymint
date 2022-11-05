#ifndef SUBSTITUTION_CUH
#define SUBSTITUTION_CUH

#include <vector>

#include "Macros.cuh"

namespace Sym {
    DECLARE_SYMBOL(Substitution, false)
    static const size_t SUBSTITUTION_NAME_COUNT;
    static const char* const SUBSTITUTION_NAMES[];

    size_t substitution_idx;

    /*
     * @brief Odwraca i kompresuje ciąg podstawień (`compress_reverse_to` robi to tylko dla
     * pojedynczego podstawienia)
     *
     * @param destination Docelowa lokalizacja podstawień
     */
    __host__ __device__ size_t compress_reverse_substitutions_to(Symbol* const destination) const;

    __host__ __device__ static void create(const Symbol* const expression,
                                           Symbol* const destination,
                                           const size_t substitution_idx);
    static std::string nth_substitution_name(const size_t n);

    __host__ __device__ Symbol* expression();
    [[nodiscard]] __host__ __device__ const Symbol* expression() const;
    [[nodiscard]] __host__ __device__ Substitution* next_substitution();
    [[nodiscard]] __host__ __device__ const Substitution* next_substitution() const;
    [[nodiscard]] __host__ __device__ bool is_last_substitution() const;

    /*
     * @brief Zamienia wszystkie wystąpienia zmiennej w wyrażeniu na nazwę z `SUBSTITUTION_NAMES`
     * zależnie od wartości `substitution_idx`
     *
     * @return Wyrażenie z zamienioną zmienną
     */
    [[nodiscard]] std::vector<Symbol> expression_with_constant() const;

    /*
     * @brief Zwraca podstawienie jako string, ale bez uwzględniania kolejnych podstawień.
     *
     * @return String przedstawiający podstawienie
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
