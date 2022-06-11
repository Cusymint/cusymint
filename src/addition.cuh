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
     * @brief Uproszczenie struktury dodawania do drzewa w postaci:
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
     * @brief W drzewie o uproszczonej strukturze wyszukuje par upraszczalnych wyrażeń.
     */
    __host__ __device__ void simplify_pairs();

    /*
     * @brief Sprawdza, czy `expr1 == sin^2(x)` i `expr2 == cos^2(x)`
     *
     * @return `true` jeśli powyższe to prawda, `false` w.p.p.
     */
    __host__ __device__ static bool is_sine_cosine_squared_sum(const Symbol* const expr1,
                                                               const Symbol* const expr2);

    /*
     * @brief Sprawdza, czy dwa drzewa można uprościć dodawaniem. Jeśli tak, to to robi
     *
     * @param expr1 Pierwszy składnik sumy
     * @param expr2 Drugi składnik sumy
     *
     * @return `true` jeśli wykonano uproszczenie, `false` w przeciwnym wypadku
     */
    __host__ __device__ static bool try_add_symbols(Symbol* const expr1, Symbol* const expr2);

    /*
     * @brief W drzewie dodawań usuwa dodawania, których jednym z argumentów jest 0.0. Dodawanie
     * takie jest zamieniane na niezerowy argument lub 0.0, jeśli oba argumenty były zerem.
     * Pozostawia drzewo z niekoherentnymi rozmiarami (wymaga wywołania compress_reverse_to).
     * UWAGA: funkcja ta może zmienić typ `this`, np. kiedy `this == + -1.0 1.0`, to po wykonaniu
     * funkcji `this == 0.0`, czyli typ się zmienił z Addition na NumericConstant! Nie należy więc
     * po wywołaniu tej funkcji wywoływać już żadnych innych funkcji składowych z Addition!
     */
    __host__ __device__ void eliminate_zeros();

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
         * @brief Przesuwa iterator do przodu.
         *
         * @return `true` jeśli przesunięty na kolejny element, `false` w przeciwnym wypadku.
         */
        __host__ __device__ bool advance();

        /*
         * @brief Zwraca informacje o ważności iteratora.
         *
         * @return `true` jeśli `current() != nullptr`, `false` w przeciwnym wypadku.
         */
        __host__ __device__ bool is_valid();

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
