#ifndef INTEGRATE_KERNELS_CUH
#define INTEGRATE_KERNELS_CUH

#include "Evaluation/Status.cuh"

#include "Symbol/ExpressionArray.cuh"

namespace Sym {
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
     * @brief Gets element signaled by inclusive_scan[index]
     *
     * @param index Index to check
     * @param inclusive_scan Array of element sizes on which inclusive_scan has been run
     *
     * @return Value of the element before inclusive_scan was run
     */
    __device__ uint32_t get_value_from_scan(const size_t index,
                                            const Util::DeviceArray<uint32_t>& inclusive_scan);
}

namespace Sym::Kernel {
    static constexpr size_t SIMPLIFY_SHARED_MEM_BYTES = 48LU * 1024LU;
    static constexpr size_t SIMPLIFY_SHARED_MEM_SYMBOLS =
        SIMPLIFY_SHARED_MEM_BYTES / sizeof(Symbol);

    /*
     * @brief Simplifies `expressions`
     *
     * @param expressions Expressions to simplify
     * @param destination Destination to save the simplified expressions
     * @param help_spaces Help space required for some simplifications
     * @param statuses Evaluation statuses used for reporting reallocation requests
     */
    __global__ void simplify(const ExpressionArray<> expressions, ExpressionArray<> destination,
                             ExpressionArray<> help_spaces,
                             Util::DeviceArray<EvaluationStatus> statuses);

    /*
     * @brief Checks whether `integrals` have known solutions
     *
     * @param integrals Integrals to be checked
     * @param applicability Solution to checking all `integrals` against known integrals.
     * `applicability[MAX_EXPRESSION_COUNT * form_idx + int_idx]` stores information whether
     * `KnownIntegral::APPLICATIONS[form_idx]` can be applied to `integral[int_idx]`, where
     * `MAX_EXPRESSION_COUNT` is the maximum size of the `integrals` array.
     */
    __global__ void
    check_for_known_integrals(const ExpressionArray<SubexpressionCandidate> integrals,
                              Util::DeviceArray<uint32_t> applicability);

    /*
     * @brief Solves integrals using the `applicability` information from
     * `check_for_known_integrals` and sets `SubexpressionCandidates` in `expressions` as solved
     * by a corresponding `SubexpressionVacancy`
     *
     * @param integrals Integrals with potentially known solutions
     * @param expressions Expressions containing SubexpressionVacancies, and also where results will
     * be saved.
     * @param dst_offset Starting index at which results will be saved.
     * @param help_spaces Help space used in applying known integrals
     * @param applicability Result of `inclusive_scan` on `check_for_known_integrals()`
     * applicability array
     * @param candidates_created Evaluation statuses for all created integrals that are set
     * according to the result of applications
     * @param candidates_created Number of `SubexpressionCandidates` already created
     * in integration process
     */
    __global__ void apply_known_integrals(const ExpressionArray<SubexpressionCandidate> integrals,
                                          ExpressionArray<> expressions, const size_t dst_offset,
                                          ExpressionArray<> help_spaces,
                                          const Util::DeviceArray<uint32_t> applicability,
                                          Util::DeviceArray<EvaluationStatus> statuses,
                                          const size_t candidates_created);

    /*
     * @brief Marks SubexpressionsVacancies as solved (sets `is_solved` and `solver_id`)
     * when there is a SubexpressionCandidate with all its SubexpressionVacancies solved.
     *
     * @param expressions Expressions to propagate information about being solved
     */
    __global__ void propagate_solved_subexpressions(ExpressionArray<> expressions);

    /*
     * @brief Finds redundant SubexpressionCandidates which are children of already solved
     * SubexpressionVacancies. SubexpressionCandidates that are solutions to
     * SubexpressionsVacancies are not marked.
     *
     * @param expressions Expressions containing redundant SubexpressionCandidates
     * @param removability Solution. `0` is set for redundant SubexpresionCandidates
     */
    __global__ void find_redundand_expressions(const ExpressionArray<> expressions,
                                               Util::DeviceArray<uint32_t> removability);

    /*
     * @brief Find integrals solving redundant SubexpressionVacancies
     *
     * @param integrals Integrals to be checked against
     * @param expressions Expressions pointing to integrals
     * @param expressions_removability Result of `find_redundand_expression()`.
     * `0` for expressions to be deleted, `1` for the rest
     * @param integrals_removability Result. `0` for redundant integrals,
     * `1` otherwise
     */
    __global__ void
    find_redundand_integrals(const ExpressionArray<> integrals, const ExpressionArray<> expressions,
                             const Util::DeviceArray<uint32_t> expressions_removability,
                             Util::DeviceArray<uint32_t> integrals_removability);

    /*
     * @brief Moves `expressions` to `destinations` skipping those marked by `removability`.
     * Updates `solver_idx` and `vacancy_expression_idx`. Zeroes `candidate_integral_count`.
     *
     * @tparam ZERO_CANDIDATE_INTEGRAL_COUNT Whether to zero `candidate_integral_count` of
     * vacancies in candidates that are moved to `destinations`
     * @param expressions Expressions to be moved
     * @param removability New locations indices of `expressions`. If `removability[i] ==
     * removability[i - 1]` or `i == 0 && removability[i] != 0` then expression is moved to
     * `destination[removability[i] - 1]`.
     * @param destinations Destination to move integrals to
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
            expressions[expr_idx].copy_to(destination);

            destination.if_is_do<SubexpressionCandidate>([&removability](auto& dst) {
                dst.vacancy_expression_idx = removability[dst.vacancy_expression_idx] - 1;
            });

            for (size_t symbol_idx = 0; symbol_idx < destination.size(); ++symbol_idx) {
                destination[symbol_idx].if_is_do<SubexpressionVacancy>([&removability](auto& vac) {
                    // We copy this value regardless of whether `vac` is really solved, if
                    // it is not, then `solver_idx` contains garbage anyways
                    vac.solver_idx = removability[vac.solver_idx] - 1;

                    if constexpr (ZERO_CANDIDATE_INTEGRAL_COUNT) {
                        vac.candidate_integral_count = 0;
                    }
                });
            }
        }
    }

    /*
     * @brief Moves `integrals` to `destinations` removing those specified by
     * `removability` and updates `vacancy_expression_idx`.
     *
     * @param integrals Integrals to be moved
     * @param integrals_removability Indexes of integrals in `destinations`.
     * When `integrals_removability[i] == integrals_removability[i - 1]`
     * or `i == 0 && removability[i] != 0` the expression is moved to
     * `destinations[removability[i] - 1]`.
     * @param destinations Place to move correct integrals to
     */
    __global__ void remove_integrals(const ExpressionArray<SubexpressionCandidate> integrals,
                                     const Util::DeviceArray<uint32_t> integrals_removability,
                                     const Util::DeviceArray<uint32_t> expressions_removability,
                                     ExpressionArray<> destinations);

    /*
     * @brief Checks which heuristics are applicable to which integrals and updates
     * `candidate_integral_count` and `candidate_expression_count` in correct expressions
     *
     * @param integrals Integrals to be checked
     * @param expressions Parents of SubexpressionCandidate in `integrals`.
     * @param new_integrals_flags Solution. When `integrals[i]` matches `Heuristic::CHECKS[j]`
     * sets `new_integrals_flags[MAX_EXPRESSION_COUNT * j + i]` to `1`, otherwise `0`.
     * @param new_expressions_flags  Solution. When `integrals[i]` matches
     * `Heuristic::CHECKS[j]` sets `new_expressions_flags[MAX_EXPRESSION_COUNT * j + i]` to `1`,
     * otherwise `0`.
     */
    __global__ void
    check_heuristics_applicability(const ExpressionArray<SubexpressionCandidate> integrals,
                                   ExpressionArray<> expressions,
                                   Util::DeviceArray<uint32_t> new_integrals_flags,
                                   Util::DeviceArray<uint32_t> new_expressions_flags);

    /*
     * @brief Applies heuristics to integrals
     *
     * @param integrals Integrals to which heuristics will be applied
     * @param integrals_dst Destinations of transformed integrals. Has to contain
     * enough expressions to contain all results.
     * @param expressions_dst Destination for new expressions. Has to contain enough
     * expressions to contain all results.
     * @param expressions_dst_offset Starting index at which result expressions will be saved
     * @param help_spaces Help space for transformations
     * @param new_integrals_indices Indices of new integrals incremented by 1.
     * If given index is equal to its predecessor, then its integral and heuristic
     * (specified in `check_heuristics_applicability()`) haven't found any solution.
     * `new_integrals_indices[0]` will override `1` to `0`.
     * @param new_expressions_indices Analogical to `new_integrals_indices` for `expressions`
     * @param integral_statuses Evaluation statuses for all created integrals that are set
     * according to the result of applications
     * @param expression_statuses Evaluation statuses for all created expressions that are set
     * according to the result of applications
     * @param candidates_created Number of `SubexpressionCandidates` already created
     * in integration process
     */
    __global__ void apply_heuristics(
        const ExpressionArray<SubexpressionCandidate> integrals, ExpressionArray<> integrals_dst,
        ExpressionArray<> expressions_dst, const size_t expressions_dst_offset,
        ExpressionArray<> help_spaces, const Util::DeviceArray<uint32_t> new_integrals_indices,
        const Util::DeviceArray<uint32_t> new_expressions_indices,
        Util::DeviceArray<EvaluationStatus> integral_statuses,
        Util::DeviceArray<EvaluationStatus> expression_statuses, const size_t candidates_created);

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
     * @param failures Arrays that should point out already failed expressions, all descendands
     * of which are going to be failed (failures[i] == 0 iff failed, 1 otherwise)
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
