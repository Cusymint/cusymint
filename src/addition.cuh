#ifndef ADDITION_CUH
#define ADDITION_CUH

#include <vector>

#include "symbol_defs.cuh"

namespace Sym {
    DECLARE_SYMBOL(Addition, false)
    TWO_ARGUMENT_OP_SYMBOL
    std::string to_string() const;

  private:
    /*
     * @brief uproszczenie struktury dodawania do drzewa w postaci:
     *                 +
     *                / \
     *               +   e
     *              / \
     *             +   d
     *            / \
     *           +   c
     *          / \
     *         a   b
     *
     * Zakładamy, że oba argumenty są w uproszczonej postaci
     *
     * @param help_space Pamięć pomocnicza
     */
    __host__ __device__ void simplify_structure(Symbol* const help_space);

    /*
     * @brief W uproszczonym drzewie dodawań zwraca dodawanie najniżej w drzewie
     *
     * @return Wskaźnik do ostatniego dodawania. Jeśli `arg1()` nie jest dodawaniem, to zwraca
     * `this`
     */
    __host__ __device__ const Addition* last_in_tree() const;

    /*
     * @brief W uproszczonym drzewie dodawań zwraca dodawanie najniżej w drzewie
     *
     * @return Wskaźnik do ostatniego dodawania. Jeśli `arg1()` nie jest dodawaniem, to zwraca
     * `this`
     */
    __host__ __device__ Addition* last_in_tree();
    END_DECLARE_SYMBOL(Addition)

    DECLARE_SYMBOL(Negation, false)
    ONE_ARGUMENT_OP_SYMBOL
    std::string to_string() const;
    END_DECLARE_SYMBOL(Negation)

    std::vector<Symbol> operator+(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs);
    std::vector<Symbol> operator-(const std::vector<Symbol>& arg);
    std::vector<Symbol> operator-(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs);
}

#endif
