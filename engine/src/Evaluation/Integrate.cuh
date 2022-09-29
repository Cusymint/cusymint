#ifndef INTEGRATE_CUH
#define INTEGRATE_CUH

#include "Symbol/ExpressionArray.cuh"

#include "Symbol/Symbol.cuh"

namespace Sym {
    using ApplicabilityCheck = size_t (*)(const Integral* const);
    using IntegralTransform = void (*)(const Integral* const, Symbol* const, Symbol* const);

    __device__ size_t is_single_variable(const Integral* const integral);
    __device__ size_t is_simple_variable_power(const Integral* const integral);
    __device__ size_t is_variable_exponent(const Integral* const integral);
    __device__ size_t is_simple_sine(const Integral* const integral);
    __device__ size_t is_simple_cosine(const Integral* const integral);
    __device__ size_t is_constant(const Integral* const integral);
    __device__ size_t is_known_arctan(const Integral* const integral);

    __device__ void integrate_single_variable(const Integral* const integral,
                                              Symbol* const destination, Symbol* const help_space);
    __device__ void integrate_simple_variable_power(const Integral* const integral,
                                                    Symbol* const destination,
                                                    Symbol* const help_space);
    __device__ void integrate_variable_exponent(const Integral* const integral,
                                                Symbol* const destination,
                                                Symbol* const help_space);
    __device__ void integrate_simple_sine(const Integral* const integral, Symbol* const destination,
                                          Symbol* const help_space);
    __device__ void integrate_simple_cosine(const Integral* const integral,
                                            Symbol* const destination, Symbol* const help_space);
    __device__ void integrate_constant(const Integral* const integral, Symbol* const destination,
                                       Symbol* const help_space);
    __device__ void integrate_arctan(const Integral* const integral, Symbol* const destination,
                                     Symbol* const help_space);

    __device__ size_t is_function_of_ex(const Integral* const integral);

    __device__ void transform_function_of_ex(const Integral* const integral,
                                             Symbol* const destination, Symbol* const help_space);

    /*
     * @brief Tworzy symbol `Solution` i zapisuje go na `destination` razem z podstawieniami z
     * `integral`
     *
     * @param integral Całka z której skopiowane mają być podstawienia
     * @param destination Miejsce do zapisania wyniku
     *
     * @return Wskaźnik na symbol za ostatnim podstawieniem
     */
    __device__ Symbol* prepare_solution(const Integral* const integral, Symbol* const destination);

    // HEURISTIC_CHECK_COUNT cannot be defined as sizeof(heurisic_checks) because
    // `heurisic_checks` is defined in translation unit associated with integrate.cu, but its
    // size has to be known in other translation units as well
    constexpr size_t KNOWN_INTEGRAL_COUNT = 7;
    constexpr size_t HEURISTIC_CHECK_COUNT = 1;
    constexpr size_t MAX_CHECK_COUNT =
        KNOWN_INTEGRAL_COUNT > HEURISTIC_CHECK_COUNT ? KNOWN_INTEGRAL_COUNT : HEURISTIC_CHECK_COUNT;
    constexpr size_t TRANSFORM_GROUP_SIZE = 32;
    constexpr size_t MAX_EXPRESSION_COUNT = 256;
    constexpr size_t EXPRESSION_MAX_SYMBOL_COUNT = 256;
    constexpr size_t APPLICABILITY_ARRAY_SIZE = MAX_CHECK_COUNT * MAX_EXPRESSION_COUNT;
    constexpr size_t INTEGRAL_ARRAY_SIZE = MAX_EXPRESSION_COUNT * EXPRESSION_MAX_SYMBOL_COUNT;

    /*
     * @brief Upraszcza wyrażenia w `expressions`. Wynik zastępuje stare wyrażenia.
     *
     * @param expressions Wyrażenia do uproszczenia
     * @param help_spaces Dodatkowa pamięć pomocnicza przy upraszczaniu wyrażeń
     */
    __global__ void simplify(ExpressionArray<> expressions, ExpressionArray<> help_spaces);

    /*
     * @brief Sprawdza, czy całki w `integrals` mają znane rozwiązania
     *
     * @param integrals Całki do sprawdzenia znanych form
     * @param applicability Wynik sprawdzania dla każdej całki w `integrals` i każdej znanej całki.
     * Informacja, czy całka pod `int_idx` ma postać o indeksie `form_idx` zapisywana jest w
     * `applicability[MAX_EXPRESSION_COUNT * form_idx + int_idx]`, gdzie MAX_EXPRESSION_COUNT jest
     * maksymalną liczbą wyrażeń dopuszczalną w `integrals`
     */
    __global__ void check_for_known_integrals(const ExpressionArray<Integral> integrals,
                                              Util::DeviceArray<size_t> applicability);

    /*
     * @brief Na podstawie informacji z `check_for_known_integrals` przekształca całki na ich
     * rozwiązania.
     *
     * @param integrals Całki o potencjalnie znanych rozwiązaniach
     * @param expressions Wyrażenia zawierające SubexpressionVacancy do których odnoszą się całki.
     * Rozwiązania całek są zapisywane w pamięci za ostatnim wyrażeniem w expressions
     * @param help_spaces Pamięć pomocnicza do wykonania przekształceń
     * @param applicability Tablica która jest wynikiem `inclusive_scan` na tablicy o tej samej
     * nazwie zwróconej przez `check_for_known_integrals`
     */
    __global__ void apply_known_integrals(const ExpressionArray<SubexpressionCandidate> integrals,
                                          ExpressionArray<> expressions,
                                          ExpressionArray<> help_spaces,
                                          const Util::DeviceArray<size_t> applicability);

    /*
     * @brief Ustawia `is_solved` i `solver_id` w SubexpressionVacancy na które wskazują
     * SubexpressionCandidate w których wszystkie SubexpressionVacancy są rozwiązane
     *
     * @param expressions Wyrażenia do propagacji informacji o rozwiązaniach
     */
    __global__ void propagate_solved_subexpressions(ExpressionArray<> expressions);

    __global__ void check_heuristics_applicability(const ExpressionArray<Integral> integrals,
                                                   Util::DeviceArray<size_t> applicability);
    __global__ void apply_heuristics(const ExpressionArray<Integral> integrals,
                                     ExpressionArray<> destinations, ExpressionArray<> help_spaces,
                                     const Util::DeviceArray<size_t> applicability);

    /*
     * @brief Sprawdza które całki wskazują na wyrażenia rozwiązane lub takie, które będą usunięte
     *
     * @param integrals Całki do sprawdzenia
     * @param expressions_removability Wyrażenia oznaczone jako do usunięcia (0 -> będzie usunięte)
     * @param integrals_removability Oznaczenie całek które można usunąć (0 -> można usunąć)
     */
    __global__ void did_integrals_expire(const ExpressionArray<SubexpressionCandidate> integrals,
                                         const size_t* const expressions_removability,
                                         size_t* const integral_removability);

    /*
     * @brief Sprawdza które całki wskazują na wyrażenia, które będą usunięte
     *
     * @param expressions Wyrażenia na które wskazują całki
     * @param integrals Całki do sprawdzenia
     * @param expressions_removability Wyrażenia oznaczone jako do usunięcia (0 -> będzie usunięte)
     * @param integrals_removability Oznaczenie całek które można usunąć (0 -> można usunąć)
     */
    __global__ void are_integrals_failed(const ExpressionArray<> expressions,
                                         const ExpressionArray<SubexpressionCandidate> integrals,
                                         const size_t* const expressions_removability,
                                         size_t* const integral_removability);

    /*
     * @brief Przenosi wyrażenia z `expressions` do `destinations` pomijając te, które wyznacza
     * `removability`. Aktualizuje też `winner_idx` i `vacancy_expression_idx`
     *
     * @param expressions Wyrażenia do przeniesienia
     * @param removability Lokalizacje wyrażeń w `destinations`. Jeśli `removability[i] ==
     * removability[i - 1]` lub `i == 0 && removability[i] != 0` to wyrażenie przenoszone jest na
     * `destinations[removability[i] - 1]`.
     * @param destinations Docelowe miejsce zapisu wyrażeń
     */
    __global__ void remove_expressions(const ExpressionArray<> expressions,
                                       const Util::DeviceArray<size_t> removability,
                                       ExpressionArray<> destinations);

    /*
     * @brief Usuwa wyrażenia z expressions wskazane przez removability oraz
     * aktualizuje wszystkie odniesienia do nich w `subexpression_candidates`.
     *
     * @param expressions Wyrażenia z których część ma być usunięta
     * @param removability inclusive scan wyrażeń które zostają, wyrażenie `expressions[i]` będzie
     * zachowane jeśli `removability[i] - removability[i - 1] != 0` lub dla `i == 0` gdy
     * `removability[0] != 0`.
     * @param subexpression_candidates Kandydaci do podstawień w podwyrażeniach w wyrażeniach w
     * `expressions`. Zakłada, że nie znajdują się w tej tablicy żadne podwyrażenia odnoszące
     * się do `expressions`
     * @param destinations Miejsce gdzie zapisane będą wyrażenia
     */
    __global__ void
    remove_expressions(ExpressionArray<> expressions, Util::DeviceArray<size_t> removability,
                       ExpressionArray<SubexpressionCandidate> subexpression_candidates,
                       ExpressionArray<> destinations);

    /*
     * @brief Przenosi całki do `destinations`, pozostawiając te wskazane w `removability`
     *
     * @param integrals Całki do przeniesienia do `destinations`
     * @param removability inclusive scan całek które zostają, całka `integrals[i]` będzie zachowana
     * jeśli `removability[i] - removability[i - 1] != 0` lub dla `i == 0` gdy `removability[0] !=
     * 0`.
     * @param expressions Wyrażenia w których SubexpressionVacancy zostaną zaktualizowane.
     */
    __global__ void remove_integrals(const ExpressionArray<SubexpressionCandidate> integrals,
                                     const Util::DeviceArray<size_t> removability,
                                     ExpressionArray<> destinations);

    /*
     * @brief Zeruje `candidate_integral_count`
     *
     * @param expressions Wyrażenia w których `candidate_integral_count` wszystkich
     * SubexpressionVacancy będzie wyzerowane
     */
    __global__ void zero_candidate_integral_count(ExpressionArray<> expressions);
}

#endif
