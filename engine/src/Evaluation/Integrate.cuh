#ifndef INTEGRATE_CUH
#define INTEGRATE_CUH

#include "Symbol/ExpressionArray.cuh"

#include "Symbol/Symbol.cuh"

namespace Sym {
    struct HeuristicCheckResult {
        __host__ __device__ HeuristicCheckResult(const size_t new_integrals,
                                                 const size_t new_expressions) :
            new_integrals(new_integrals), new_expressions(new_expressions) {}

        size_t new_integrals;
        size_t new_expressions;
    };

    using HeuristicCheck = HeuristicCheckResult (*)(const Integral* const);
    using HeuristicApplication = void (*)(const Integral* const, Symbol* const, Symbol* const,
                                          Symbol* const);

    using KnownIntegralCheck = size_t (*)(const Integral* const);
    using KnownIntegralTransform = void (*)(const Integral* const, Symbol* const, Symbol* const);

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

    __device__ HeuristicCheckResult is_function_of_ex(const Integral* const integral);

    __device__ void transform_function_of_ex(const Integral* const integral,
                                             Symbol* const integral_dst,
                                             Symbol* const expression_dst,
                                             Symbol* const help_space);

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
    constexpr size_t SCAN_ARRAY_SIZE = MAX_CHECK_COUNT * MAX_EXPRESSION_COUNT;
    constexpr size_t INTEGRAL_ARRAY_SIZE = MAX_EXPRESSION_COUNT * EXPRESSION_MAX_SYMBOL_COUNT;

    /*
     * @brief Checks if inclusive_scan[index] is signaling a zero-sized element
     *
     * @param index Index to check
     * @param inclusive_scan Array of element sizes on which inclusive_scan has been run
     *
     * @return `false` if element is zero-sized, `true` otherwise
     */
    __device__ bool is_nonzero(const size_t index,
                               const Util::DeviceArray<uint32_t>& inclusive_scan);

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
    __global__ void
    check_for_known_integrals(const ExpressionArray<SubexpressionCandidate> integrals,
                              Util::DeviceArray<uint32_t> applicability);

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
                                          const Util::DeviceArray<uint32_t> applicability);

    /*
     * @brief Ustawia `is_solved` i `solver_id` w SubexpressionVacancy na które wskazują
     * SubexpressionCandidate w których wszystkie SubexpressionVacancy są rozwiązane
     *
     * @param expressions Wyrażenia do propagacji informacji o rozwiązaniach
     */
    __global__ void propagate_solved_subexpressions(ExpressionArray<> expressions);

    /*
     * @brief Oznacza SubexpressionCandidate wskazujące na SubexpressionVacancy rozwiązane przez
     * innych kandydatów
     *
     * @param expressions Wyrażenia do sprawdzenia
     * @param removability Wynik sprawdzania. `0` dla wyrażeń, dla których istnieje przodek
     * rozwiązany w innej linii.
     */
    __global__ void find_redundand_expressions(const ExpressionArray<> expressions,
                                               Util::DeviceArray<uint32_t> removability);

    /*
     * @brief Oznacza całki wskazujące na wyrażenia, które będą usunięte lub wskazujące na
     * SubexpressionVacancy, które są rozwiązane
     *
     * @param integrals Całki do sprawdzenia
     * @param expressions Wyrażenia, na które wskazują całki
     * @param expressions_removability Wyrażenia, które mają być usunięte. `0` dla tych, które
     * zostaną usunięte, `1` dla pozostałych
     * @param integrals_removability Wynik sprawdzania. `0` dla całek, które wskazują na rozwiązane
     * podwyrażenia, `1` w przeciwnym wypadku
     */
    __global__ void
    find_redundand_integrals(const ExpressionArray<> integrals, const ExpressionArray<> expressions,
                             const Util::DeviceArray<uint32_t> expressions_removability,
                             Util::DeviceArray<uint32_t> integrals_removability);

    /*
     * @brief Przenosi wyrażenia z `expressions` do `destinations` pomijając te, które wyznacza
     * `removability`. Aktualizuje też `solver_idx` i `vacancy_expression_idx`, oraz zeruje
     * `candidate_integral_count` (przygotowanie do sprawdzania heurystyk)
     *
     * @param expressions Wyrażenia do przeniesienia
     * @param removability Lokalizacje wyrażeń w `destinations`. Jeśli `removability[i] ==
     * removability[i - 1]` lub `i == 0 && removability[i] != 0` to wyrażenie przenoszone jest na
     * `destinations[removability[i] - 1]`.
     * @param destinations Docelowe miejsce zapisu wyrażeń
     */
    template <bool ZERO_CANDIDATE_INTEGRAL_COUNT = false>
    __global__ void remove_expressions(const ExpressionArray<> expressions,
                                       const Util::DeviceArray<uint32_t> removability,
                                       ExpressionArray<> destinations) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        for (size_t expr_idx = thread_idx; expr_idx < expressions.size();
             expr_idx += thread_count) {
            if (!is_nonzero(expr_idx, removability)) {
                continue;
            }

            Symbol& destination = destinations[removability[expr_idx] - 1];
            expressions[expr_idx].copy_to(&destination);

            destination.if_is_do<SubexpressionCandidate>([&removability](auto& dst) {
                dst.vacancy_expression_idx = removability[dst.vacancy_expression_idx] - 1;
            });

            for (size_t symbol_idx = 0; symbol_idx < destination.size(); ++symbol_idx) {
                destination[symbol_idx].if_is_do<SubexpressionVacancy>([&removability](auto& vac) {
                    // We copy this value regardless of whether `vac` is really solved, if it is
                    // not, then `solver_idx` contains garbage anyways
                    vac.solver_idx = removability[vac.solver_idx] - 1;

                    if constexpr (ZERO_CANDIDATE_INTEGRAL_COUNT) {
                        vac.candidate_integral_count = 0;
                    }
                });
            }
        }
    }

    /*
     * @brief Przenosi całki z `integrals` do `destinations` pomijając te, które wyznacza
     * `removability`. Aktualizuje też `vacancy_expression_idx`.
     *
     * @param integrals Całki do przeniesienia
     * @param integrals_removability Lokalizacje wyrażeń w `destinations`. Jeśli
     * `integrals_removability[i] == integrals_removability[i - 1]` lub `i == 0 && removability[i]
     * != 0` to wyrażenie przenoszone jest na `destinations[removability[i] - 1]`.
     * @param expressions_removability To samo co `integrals_removability`, tylko że dla wyrażeń na
     * które wskazuję SubexpressionCandidate w `integrals`
     * @param destinations Docelowe miejsce zapisu całek
     */
    __global__ void remove_integrals(const ExpressionArray<SubexpressionCandidate> integrals,
                                     const Util::DeviceArray<uint32_t> integrals_removability,
                                     const Util::DeviceArray<uint32_t> expressions_removability,
                                     ExpressionArray<SubexpressionCandidate> destinations);

    /*
     * @brief Sprawdza, które heurystyki pasują do których całek oraz aktualizuje
     * `candidate_integral_count` i `candidate_expression_count` w wyrażeniach, na które wskazywać
     * będą utworzone później całki.
     *
     * @param integrals Całki do sprawdzenia
     * @param expressions Wyrażenia na które wskazują SubexpressionCandidate w `integrals`.
     * @param new_integrals_flags Jeśli całka na indeksie `i` pasuje do heurystyki na indeksie `j`,
     * która przekształca na inną całkę, to new_integrals_flags[MAX_EXPRESSION_COUNT * j + i] będzie
     * ustawione na `1`.
     * @param new_expressions_flags Jeśli całka na indeksie `i` pasuje do heurystyki na indeksie
     * `j`, która przekształca ją na wyrażenie z całek, to
     * `new_expressions_flags[MAX_EXPRESSION_COUNT * j + i]` będzie ustawione na `1`.
     */
    __global__ void
    check_heuristics_applicability(const ExpressionArray<SubexpressionCandidate> integrals,
                                   ExpressionArray<> expressions,
                                   Util::DeviceArray<uint32_t> new_integrals_flags,
                                   Util::DeviceArray<uint32_t> new_expressions_flags);

    /*
     * @brief Stosuje heurystyki na całkach
     *
     * @param integrals Całki, na których zastosowane zostaną heurystyki
     * @param integrals_destinations Miejsce, gdzie zapisane będą nowe całki
     * @param expressions_destinations Miejsce, gdzie zapisane będą nowe wyrażenia. Wyrażenia
     * zapisywane są od pierwszego niezajętego miesca za końcem tablicy (czyli obecna zawartość jest
     * nienaruszona).
     * @param help_spaces Pamięć pomocnicza do wykonywania przekształceń
     * @param new_integrals_indices Indeksy nowych całek powiększone o 1. Jeśli jakiś indeks jest
     * równy poprzedniemu, to wskazywane przez niego połączenie całki i heurystyki (indeksowanie
     * opisane przy `check_heuristics_applicability`) nie dało żadnego wyniku. Dla
     * `new_integrals_indices[0]` jest to sygnalizowane przez zapisaną tam wartość 0 (zamiast 1)
     * @param new_expressions_indices Inteksy nowych wyrażeń powiększone o 1. Zasady takie same jak
     * w `new_integrals_indices`
     */
    __global__ void apply_heuristics(const ExpressionArray<SubexpressionCandidate> integrals,
                                     ExpressionArray<> integrals_destinations,
                                     ExpressionArray<> expressions_destinations,
                                     ExpressionArray<> help_spaces,
                                     const Util::DeviceArray<uint32_t> new_integrals_indices,
                                     const Util::DeviceArray<uint32_t> new_expressions_indices);

    /*
     * @brief Propagates information about failed SubexpressionVacancy upwards to parent
     * expressions.
     *
     * @param expressions Expressions to update
     * @param failures Array that should be filled with values of `1`, if `expressions[i]` fails
     * then `failures[i]` is set to 0
     */
    __global__ void propagate_failures_upwards(ExpressionArray<> expressions,
                                               Util::DeviceArray<uint32_t> failures);

    /*
     * @brief Propagates information about failed SubexpressionCandidate downwards
     *
     * @param expression Expressions to update
     * @param failurs Arrays that should point out already failed expressions, all descendands of
     * which are going to be failed (failures[i] == 0 iff failed, 1 otherwise)
     */
    __global__ void propagate_failures_downwards(ExpressionArray<> expressions,
                                                 Util::DeviceArray<uint32_t> failures);

    /*
     * @brief Marks integrals that point to expressions which are going to be removed
     *
     * @param integrals Integrals to mark
     * @param expressions_removability Expressions which are going to be removed, 0 for the ones
     * awaiting removal, 1 for the ones staying
     * @param integrals_removability Checking result, same convention as in
     * `expressions_removability`
     */
    __global__ void
    find_redundand_integrals(const ExpressionArray<> integrals,
                             const Util::DeviceArray<uint32_t> expressions_removability,
                             Util::DeviceArray<uint32_t> integrals_removability);
}

#endif
