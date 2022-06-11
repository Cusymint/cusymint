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

    /*
     * @brief Iterator kolejnych wyrażeń w uproszczonym dodawaniu.
     */
    class AdditionIterator {
        Addition* current_addition;
        Symbol* current_symbol;

      public:
        /*
         * @brief Tworzy iterator dla `addition`
         *
         * @param addition Symbol po którego dzieciach miejsce będzie mieć iteracja
         */
        __host__ __device__ AdditionIterator(Addition* const addition);

        /*
         * @brief Przesuwa iterator do przodu
         *
         * @return `true` jeśli przesunięty na kolejny element, `false` w przeciwnym wypadku
         */
        __host__ __device__ bool advance();

        /*
         * @brief Zwraca obecny element
         *
         * @return Element na który obecnie wskazuje iterator. Zwraca `nullptr` gdy koniec.
         */
        __host__ __device__ Symbol* current();
    };

    DECLARE_SYMBOL(Negation, false)
    ONE_ARGUMENT_OP_SYMBOL
    std::string to_string() const;
    END_DECLARE_SYMBOL(Negation)

    std::vector<Symbol> operator+(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs);
    std::vector<Symbol> operator-(const std::vector<Symbol>& arg);
    std::vector<Symbol> operator-(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs);
}

#endif
