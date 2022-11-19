#include "Integrate.cuh"

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "Heuristic/Heuristic.cuh"
#include "KnownIntegral/KnownIntegral.cuh"
#include "StaticFunctions.cuh"

#include "Utils/CompileConstants.cuh"
#include "Utils/Cuda.cuh"
#include "Utils/Meta.cuh"

namespace Sym {
    namespace {
        constexpr size_t TRANSFORM_GROUP_SIZE = 32;
        constexpr size_t MAX_EXPRESSION_COUNT = 128;

        /*
         * @brief Try to set `expressions[potential_solver_idx]` (SubexpressionCandidate)
         * as a solution to its SubexpressionVacancy
         *
         * @param expressions Expressions array with a candidate to solve and a missing subexpression
         * @param potential_solver_idx Index of the potential solver
         *
         * @return `false` when haven't managed to set chosen candidate as a solution to
         * the subexpression or whetether there are still unsolved subexpressions in the parent.
         * `true` when managed to set chosen candidate as a solution and parent doesn't have any
         * unsolved subexpressions left. 
         */
        __device__ bool try_set_solver_idx(Sym::ExpressionArray<>& expressions,
                                           const size_t potential_solver_idx) {
            const size_t& vacancy_expr_idx =
                expressions[potential_solver_idx].subexpression_candidate.vacancy_expression_idx;

            const size_t& vacancy_idx =
                expressions[potential_solver_idx].subexpression_candidate.vacancy_idx;

            Sym::SubexpressionVacancy& subexpr_vacancy =
                expressions[vacancy_expr_idx][vacancy_idx].subexpression_vacancy;

            const bool solver_lock_acquired = atomicCAS(&subexpr_vacancy.is_solved, 0, 1) == 0;

            if (!solver_lock_acquired) {
                return false;
            }

            subexpr_vacancy.solver_idx = potential_solver_idx;

            if (!expressions[vacancy_expr_idx].is(Sym::Type::SubexpressionCandidate)) {
                return true;
            }

            unsigned int subexpressions_left = atomicSub(
                &expressions[vacancy_expr_idx].subexpression_candidate.subexpressions_left, 1);

            return subexpressions_left == 0;
        }

        /*
         * @brief Sets `var` to `val` atomically
         *
         * @brief var Variable to set
         * @brief val Value assigned to `var`
         *
         * @return `false` if `var` was already equal to `val`, `true` otherwise
         */
        template <class T> __device__ bool try_set(T& var, const T& val) {
            const unsigned int previous_val = atomicExch(&var, val);
            return previous_val != val;
        }

        /*
         * @brief Gets target index from `scan` inclusive scan array at `index` index
         */
        __device__ uint32_t index_from_scan(const Util::DeviceArray<uint32_t>& scan,
                                            const size_t index) {
            if (index == 0) {
                return 0;
            }

            return scan[index - 1];
        }

        /*
         * @brief Checks if inclusive_scan[index] is signaling a zero-sized element
         *
         * @param index Index to check
         * @param inclusive_scan Array of element sizes on which inclusive_scan has been run
         *
         * @return `false` if element is zero-sized, `true` otherwise
         */
        __device__ bool is_nonzero(const size_t index,
                                   const Util::DeviceArray<uint32_t>& inclusive_scan) {
            return index == 0 && inclusive_scan[index] != 0 ||
                   index != 0 && inclusive_scan[index - 1] != inclusive_scan[index];
        }

        /*
         * @brief Simplifies `expressions`. Result overrides `expressions` data.
         *
         * @param expressions Expressions to simplify
         * @param help_spaces Help space required for some simplifications
         */
        __global__ void simplify(ExpressionArray<> expressions, ExpressionArray<> help_spaces) {
            const size_t thread_count = Util::thread_count();
            const size_t thread_idx = Util::thread_idx();

            for (size_t expr_idx = thread_idx; expr_idx < expressions.size();
                 expr_idx += thread_count) {
                expressions[expr_idx].simplify(help_spaces.at(expr_idx));
            }
        }

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
                                  Util::DeviceArray<uint32_t> applicability) {
            const size_t thread_count = Util::thread_count();
            const size_t thread_idx = Util::thread_idx();

            const size_t check_step = thread_count / TRANSFORM_GROUP_SIZE;

            for (size_t check_idx = thread_idx / TRANSFORM_GROUP_SIZE;
                 check_idx < KnownIntegral::COUNT; check_idx += check_step) {
                for (size_t int_idx = thread_idx % TRANSFORM_GROUP_SIZE; int_idx < integrals.size();
                     int_idx += TRANSFORM_GROUP_SIZE) {
                    size_t appl_idx = MAX_EXPRESSION_COUNT * check_idx + int_idx;
                    applicability[appl_idx] =
                        KnownIntegral::CHECKS[check_idx](integrals[int_idx].arg().as<Integral>());
                }
            }
        }

        /*
         * @brief Solves integrals in place using the `applicability` information from `check_for_known_integrals`
         *
         * @param integrals Integrals with potentially known solutions
         * @param expressions Expressions containing SubexpressionVacancies.
         * Solutions are written after the last expression in `expressions`.
         * 
         * @param help_spaces Help space used in applying known integrals
         * @param applicability Result of `inclusive_scan` on `check_for_known_integrals()` applicability array
         */
        __global__ void
        apply_known_integrals(const ExpressionArray<SubexpressionCandidate> integrals,
                              ExpressionArray<> expressions, ExpressionArray<> help_spaces,
                              const Util::DeviceArray<uint32_t> applicability) {
            const size_t thread_count = Util::thread_count();
            const size_t thread_idx = Util::thread_idx();

            const size_t trans_step = thread_count / TRANSFORM_GROUP_SIZE;

            for (size_t trans_idx = thread_idx / TRANSFORM_GROUP_SIZE;
                 trans_idx < KnownIntegral::COUNT; trans_idx += trans_step) {
                for (size_t int_idx = thread_idx % TRANSFORM_GROUP_SIZE; int_idx < integrals.size();
                     int_idx += TRANSFORM_GROUP_SIZE) {
                    const size_t appl_idx = MAX_EXPRESSION_COUNT * trans_idx + int_idx;

                    if (!is_nonzero(appl_idx, applicability)) {
                        continue;
                    }

                    const size_t dest_idx =
                        expressions.size() + index_from_scan(applicability, appl_idx);

                    auto* const subexpr_candidate = expressions.at(dest_idx)
                                                    << SubexpressionCandidate::builder();
                    subexpr_candidate->copy_metadata_from(integrals[int_idx]);
                    KnownIntegral::APPLICATIONS[trans_idx](integrals[int_idx].arg().as<Integral>(),
                                                           subexpr_candidate->arg(),
                                                           help_spaces[dest_idx]);
                    subexpr_candidate->seal();

                    try_set_solver_idx(expressions, dest_idx);
                }
            }
        }

        /*
         * @brief Marks SubexpressionsVacancies as solved (sets `is_solved` and `solver_id`)
         * when there is a SubexpressionCandidate with all its SubexpressionVacancies solved.
         *
         * @param expressions Expressions to propagate information about being solved
         */
        __global__ void propagate_solved_subexpressions(ExpressionArray<> expressions) {
            const size_t thread_count = Util::thread_count();
            const size_t thread_idx = Util::thread_idx();

            // For each tree node there is a seperate starting thread.
            // If its node is solved it moves to it's parent.
            // It tries to fill the parent's vacancy with it's own solution.
            // If it succeeds and all of the parent's vacancies are solved, it moves to the parent.
            // This operation upwards is repeated upwards while all solutions to the current node exists
            // and the parent's vacancy is not solved and ends at the root.

            // Since `expr_idx = 0` is SubexpressionVacancy of the original integral, it is skipped
            for (size_t expr_idx = thread_idx + 1; expr_idx < expressions.size();
                 expr_idx += thread_count) {
                size_t current_expr_idx = expr_idx;
                while (current_expr_idx != 0) {
                    if (expressions[current_expr_idx].subexpression_candidate.subexpressions_left !=
                        0) {
                        break;
                    }

                    if (!try_set_solver_idx(expressions, current_expr_idx)) {
                        break;
                    }

                    // We iterate tree upwards.
                    // It may seem that there is a possibility of race condition
                    // when we will reach the same node, as the thread which has started the loop.
                    // However, since `try_set_solver_idx` is atomic, only one thread would be able
                    // to set `solver_idx` on the next parent and continue its journey upwards.
                    current_expr_idx = expressions[current_expr_idx]
                                           .subexpression_candidate.vacancy_expression_idx;
                }
            }
        }

        /*
         * @brief Finds redundant SubexpressionCandidates which are children of already solved SubexpressionVacancies.
         * SubexpressionCandidates that are solutions to SubexpressionsVacancies are not marked.
         *
         * @param expressions Expressions containing redundant SubexpressionCandidates
         * @param removability Solution. `0` is set for redundant SubexpresionCandidates
         */
        __global__ void find_redundand_expressions(const ExpressionArray<> expressions,
                                                   Util::DeviceArray<uint32_t> removability) {
            const size_t thread_count = Util::thread_count();
            const size_t thread_idx = Util::thread_idx();

            // Look further and further in the dependency tree and check whether we are not trying
            // to solve something that has been solved already
            for (size_t expr_idx = thread_idx; expr_idx < expressions.size();
                 expr_idx += thread_count) {
                removability[expr_idx] = 1;
                size_t current_expr_idx = expr_idx;

                while (current_expr_idx != 0) {
                    const size_t& parent_idx = expressions[current_expr_idx]
                                                   .subexpression_candidate.vacancy_expression_idx;
                    const size_t& parent_vacancy_idx =
                        expressions[current_expr_idx].subexpression_candidate.vacancy_idx;
                    const SubexpressionVacancy& parent_vacancy =
                        expressions[parent_idx][parent_vacancy_idx].subexpression_vacancy;

                    if (parent_vacancy.is_solved == 1 &&
                        parent_vacancy.solver_idx != current_expr_idx) {
                        removability[expr_idx] = 0;
                        break;
                    }

                    current_expr_idx = parent_idx;
                }
            }
        }

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
        find_redundand_integrals(const ExpressionArray<> integrals,
                                 const ExpressionArray<> expressions,
                                 const Util::DeviceArray<uint32_t> expressions_removability,
                                 Util::DeviceArray<uint32_t> integrals_removability) {
            const size_t thread_count = Util::thread_count();
            const size_t thread_idx = Util::thread_idx();

            for (size_t int_idx = thread_idx; int_idx < integrals.size(); int_idx += thread_count) {
                const size_t& vacancy_expr_idx =
                    integrals[int_idx].subexpression_candidate.vacancy_expression_idx;
                const size_t& vacancy_idx = integrals[int_idx].subexpression_candidate.vacancy_idx;

                const bool parent_expr_failed = expressions_removability[vacancy_expr_idx] == 0;
                const bool parent_vacancy_solved =
                    expressions[vacancy_expr_idx][vacancy_idx].subexpression_vacancy.is_solved == 1;

                integrals_removability[int_idx] =
                    parent_expr_failed || parent_vacancy_solved ? 0 : 1;
            }
        }

        /*
         * @brief Moves `expressions` to `destinations` skipping those marked by `removability`.
         * Updates `solver_idx` and `vacancy_expression_idx`. Zeroes `candidate_integral_count`.
         *
         * @tparam ZERO_CANDIDATE_INTEGRAL_COUNT Whether to zero `candidate_integral_count` of
         * candidates that are moved to `destinations`
         * @param expressions Expressions to be moved
         * @param removability New locations indices of `expressions`. If `removability[i] == removability[i - 1]`
         * or `i == 0 && removability[i] != 0` then expression is moved to `destination[removability[i] - 1]`. 
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
                expressions[expr_idx].copy_to(&destination);

                destination.if_is_do<SubexpressionCandidate>([&removability](auto& dst) {
                    dst.vacancy_expression_idx = removability[dst.vacancy_expression_idx] - 1;
                });

                for (size_t symbol_idx = 0; symbol_idx < destination.size(); ++symbol_idx) {
                    destination[symbol_idx].if_is_do<SubexpressionVacancy>(
                        [&removability](auto& vac) {
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
                                         ExpressionArray<> destinations) {
            const size_t thread_count = Util::thread_count();
            const size_t thread_idx = Util::thread_idx();

            for (size_t int_idx = thread_idx; int_idx < integrals.size(); int_idx += thread_count) {
                if (!is_nonzero(int_idx, integrals_removability)) {
                    continue;
                }

                Symbol& destination = destinations[integrals_removability[int_idx] - 1];
                integrals[int_idx].symbol()->copy_to(&destination);

                size_t& vacancy_expr_idx =
                    destination.as<SubexpressionCandidate>().vacancy_expression_idx;
                vacancy_expr_idx = expressions_removability[vacancy_expr_idx] - 1;
            }
        }

        /*
         * @brief Checks which heuristics are applicable to which integrals and updates
         * `candidate_integral_count` and `candidate_expression_count` in correct expressions 
         *
         * @param integrals Integrals to be checked
         * @param expressions Parents of SubexpressionCandidate in `integrals`.
         * @param new_integrals_flags Solution. When `integrals[i]` matches `Heuristic::CHECKS[j]`
         * sets `new_integrals_flags[MAX_EXPRESSION_COUNT * j + i]` to `1`, otherwise `0`. 
         * @param new_expressions_flags  Solution. When `integrals[i]` matches `Heuristic::CHECKS[j]`
         * sets `new_expressions_flags[MAX_EXPRESSION_COUNT * j + i]` to `1`, otherwise `0`. 
         */
        __global__ void
        check_heuristics_applicability(const ExpressionArray<SubexpressionCandidate> integrals,
                                       ExpressionArray<> expressions,
                                       Util::DeviceArray<uint32_t> new_integrals_flags,
                                       Util::DeviceArray<uint32_t> new_expressions_flags) {
            const size_t thread_count = Util::thread_count();
            const size_t thread_idx = Util::thread_idx();

            const size_t check_step = thread_count / TRANSFORM_GROUP_SIZE;

            for (size_t check_idx = thread_idx / TRANSFORM_GROUP_SIZE; check_idx < Heuristic::COUNT;
                 check_idx += check_step) {
                for (size_t int_idx = thread_idx % TRANSFORM_GROUP_SIZE; int_idx < integrals.size();
                     int_idx += TRANSFORM_GROUP_SIZE) {
                    size_t appl_idx = MAX_EXPRESSION_COUNT * check_idx + int_idx;
                    Heuristic::CheckResult result =
                        Heuristic::CHECKS[check_idx](integrals[int_idx].arg().as<Integral>());
                    new_integrals_flags[appl_idx] = result.new_integrals;
                    new_expressions_flags[appl_idx] = result.new_expressions;

                    const size_t& vacancy_expr_idx = integrals[int_idx].vacancy_expression_idx;
                    const size_t& vacancy_idx = integrals[int_idx].vacancy_idx;
                    SubexpressionVacancy& parent_vacancy =
                        expressions[vacancy_expr_idx][vacancy_idx].subexpression_vacancy;

                    if (result.new_expressions == 0) {
                        // Assume new integrals are direct children of the vacancy
                        atomicAdd(&parent_vacancy.candidate_integral_count, result.new_integrals);
                    }
                    else {
                        // Assume new integrals are going to be children of new expressions, which
                        // are going to be children of the vacancy
                        atomicAdd(&parent_vacancy.candidate_expression_count,
                                  result.new_expressions);
                    }
                }
            }
        }

        /*
         * @brief Applies heuristics to integrals
         *
         * @param integrals Integrals on which heuristics will be applied
         * @param integrals_destinations Solutions destination
         * @param expressions_destinations Destination for new expressions. 
         * New expressions will be appended to already existing ones.
         * @param help_spaces Help space for transformations
         * @param new_integrals_indices Indices of new integrals incremented by 1.
         * If given index is equal to its predecessor, then its integral and heuristic
         * (specified in `check_heuristics_applicability()`) haven't found any solution.
         * `new_integrals_indices[0]` will override `1` to `0`.
         * @param new_expressions_indices Analogical to `new_integrals_indices` for `expressions`
         */
        __global__ void
        apply_heuristics(const ExpressionArray<SubexpressionCandidate> integrals,
                         ExpressionArray<> integrals_destinations,
                         ExpressionArray<> expressions_destinations, ExpressionArray<> help_spaces,
                         const Util::DeviceArray<uint32_t> new_integrals_indices,
                         const Util::DeviceArray<uint32_t> new_expressions_indices) {
            const size_t thread_count = Util::thread_count();
            const size_t thread_idx = Util::thread_idx();

            const size_t trans_step = thread_count / TRANSFORM_GROUP_SIZE;

            for (size_t trans_idx = thread_idx / TRANSFORM_GROUP_SIZE; trans_idx < Heuristic::COUNT;
                 trans_idx += trans_step) {
                for (size_t int_idx = thread_idx % TRANSFORM_GROUP_SIZE; int_idx < integrals.size();
                     int_idx += TRANSFORM_GROUP_SIZE) {
                    const size_t appl_idx = MAX_EXPRESSION_COUNT * trans_idx + int_idx;
                    if (!is_nonzero(appl_idx, new_integrals_indices)) {
                        continue;
                    }

                    const size_t int_dst_idx = index_from_scan(new_integrals_indices, appl_idx);

                    if (new_expressions_indices[appl_idx] != 0) {
                        const size_t expr_dst_idx =
                            expressions_destinations.size() +
                            index_from_scan(new_expressions_indices, appl_idx);
                        Heuristic::APPLICATIONS[trans_idx](
                            integrals[int_idx], integrals_destinations.iterator(int_dst_idx),
                            expressions_destinations.iterator(expr_dst_idx),
                            help_spaces[int_dst_idx]);
                    }
                    else {
                        Heuristic::APPLICATIONS[trans_idx](
                            integrals[int_idx], integrals_destinations.iterator(int_dst_idx),
                            ExpressionArray<>::Iterator::null(), help_spaces[int_dst_idx]);
                    }
                }
            }
        }

        /*
         * @brief Propagates information about failed SubexpressionVacancy upwards to parent
         * expressions.
         *
         * @param expressions Expressions to update
         * @param failures Array that should be filled with values of `1`, if `expressions[i]` fails
         * then `failures[i]` is set to 0
         */
        __global__ void propagate_failures_upwards(ExpressionArray<> expressions,
                                                   Util::DeviceArray<uint32_t> failures) {
            const size_t thread_count = Util::thread_count();
            const size_t thread_idx = Util::thread_idx();

            for (size_t expr_idx = thread_idx; expr_idx < expressions.size();
                 expr_idx += thread_count) {
                SubexpressionCandidate& self_candidate =
                    expressions[expr_idx].subexpression_candidate;

                // Some other thread was here already, as `failures` starts with 1 everywhere
                if (failures[expr_idx] == 0) {
                    continue;
                }

                bool is_failed = false;

                // expressions[current_expr_idx][0] is subexpression_candidate, so it could be
                // skipped, but if `expr_idx == 0` it is the only SubexpressionVacancy
                for (size_t sym_idx = 0; sym_idx < expressions[expr_idx].size(); ++sym_idx) {
                    if (!expressions[expr_idx][sym_idx].is(Type::SubexpressionVacancy)) {
                        continue;
                    }

                    SubexpressionVacancy& vacancy =
                        expressions[expr_idx][sym_idx].subexpression_vacancy;

                    if (vacancy.candidate_integral_count == 0 &&
                        vacancy.candidate_expression_count == 0 && vacancy.is_solved == 0) {
                        is_failed = true;
                        break;
                    }
                }

                if (!is_failed || !try_set(failures[expr_idx], 0U)) {
                    continue;
                }

                size_t current_expr_idx = expr_idx;
                while (current_expr_idx != 0) {
                    const size_t& parent_idx = expressions[current_expr_idx]
                                                   .subexpression_candidate.vacancy_expression_idx;
                    const size_t& vacancy_idx =
                        expressions[current_expr_idx].subexpression_candidate.vacancy_idx;
                    SubexpressionVacancy& parent_vacancy =
                        expressions[parent_idx][vacancy_idx].subexpression_vacancy;

                    if (parent_vacancy.candidate_integral_count != 0 ||
                        parent_vacancy.is_solved == 1) {
                        break;
                    }

                    const size_t parent_vacancy_candidates_left =
                        atomicSub(&parent_vacancy.candidate_expression_count, 1) - 1;

                    // Go upwards if parent is failed
                    if (parent_vacancy_candidates_left != 0 || !try_set(failures[parent_idx], 0U)) {
                        break;
                    }

                    current_expr_idx = parent_idx;
                }
            }
        }

        /*
         * @brief Propagates information about failed SubexpressionCandidate downwards
         *
         * @param expression Expressions to update
         * @param failurs Arrays that should point out already failed expressions, all descendands
         * of which are going to be failed (failures[i] == 0 iff failed, 1 otherwise)
         */
        __global__ void propagate_failures_downwards(ExpressionArray<> expressions,
                                                     Util::DeviceArray<uint32_t> failures) {
            const size_t thread_count = Util::thread_count();
            const size_t thread_idx = Util::thread_idx();

            // Top expression has no parents, so we skip it
            for (size_t expr_idx = thread_idx + 1; expr_idx < expressions.size();
                 expr_idx += thread_count) {
                size_t current_expr_idx = expr_idx;

                while (current_expr_idx != 0) {
                    const size_t& parent_idx = expressions[current_expr_idx]
                                                   .subexpression_candidate.vacancy_expression_idx;

                    if (failures[parent_idx] == 0) {
                        failures[expr_idx] = 0;
                        break;
                    }

                    current_expr_idx = parent_idx;
                }
            }
        }

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
                                 Util::DeviceArray<uint32_t> integrals_removability) {
            const size_t thread_count = Util::thread_count();
            const size_t thread_idx = Util::thread_idx();

            for (size_t int_idx = thread_idx; int_idx < integrals.size(); int_idx += thread_count) {
                const size_t& parent_idx =
                    integrals[int_idx].subexpression_candidate.vacancy_expression_idx;

                integrals_removability[int_idx] = expressions_removability[parent_idx];
            }
        }

        /*
         * @brief Replaces nth symbol in `expression` with `tree`, skipping the first element of
         * `tree` and expanding substitutions if `Solution` is the second symbol in `tree`
         *
         * @param expression Expression to make the replacement in
         * @param n Index of symbol to replace
         * @param tree Expression to make replacement with. Its first symbol is skipped (assumed to
         * be SubexpressionCandidate)
         *
         * @return Copy of `expression` with the replacement
         */
        std::vector<Sym::Symbol> replace_nth_with_tree(std::vector<Sym::Symbol> expression,
                                                       const size_t n,
                                                       const std::vector<Sym::Symbol>& tree) {
            if constexpr (Consts::DEBUG) {
                if (!tree[0].is(Type::SubexpressionCandidate)) {
                    Util::crash(
                        "Invalid first symbol of tree: %s, should be SubexpressionCandidate",
                        type_name(tree[0].type()));
                }
            }

            std::vector<Sym::Symbol> tree_content;

            if (tree[1].is(Sym::Type::Solution)) {
                tree_content = tree[1].as<Sym::Solution>().substitute_substitutions();
            }
            else {
                tree_content.resize(tree.size() - 1);
                std::copy(tree.begin() + 1, tree.end(), tree_content.begin());
            }

            expression[n].init_from(Sym::ExpanderPlaceholder::with_size(tree_content.size()));

            std::vector<Sym::Symbol> new_tree(expression.size() + tree_content.size() - 1);
            expression.data()->compress_to(*new_tree.data());

            std::copy(tree_content.begin(), tree_content.end(),
                      new_tree.begin() + static_cast<int64_t>(n));

            return new_tree;
        }

        /*
         * @brief Collapses a tree of expressions with Solutions with Substitutions and
         * interreferencing SubexpressionCandidates and SubexpressionVacancies to a single
         * expression.
         *
         * @param tree Tree to collapse
         * @param n Index of tree node serving as tree root
         *
         * @return Collapsed tree
         */
        std::vector<Sym::Symbol> collapse_nth(const std::vector<std::vector<Sym::Symbol>>& tree,
                                              const size_t n) {
            std::vector<Sym::Symbol> current_collapse = tree[n];

            for (size_t i = 0; i < current_collapse.size(); ++i) {
                if (!current_collapse[i].is(Sym::Type::SubexpressionVacancy)) {
                    continue;
                }

                const auto subtree = collapse_nth(
                    tree, current_collapse[i].as<Sym::SubexpressionVacancy>().solver_idx);

                auto new_collapse = replace_nth_with_tree(current_collapse, i, subtree);
                i += new_collapse.size() - current_collapse.size();
                current_collapse = new_collapse;
            }

            return current_collapse;
        }

        /*
         * @brief Collapses a tree of expressions with Solutions with Substitutions and
         * interreferencing SubexpressionCandidates and SubexpressionVacancies to a single
         * expression
         *
         * @param tree Tree to collapse
         *
         * @return Collapsed tree
         */
        std::vector<Sym::Symbol> collapse(const std::vector<std::vector<Sym::Symbol>>& tree) {
            auto collapsed = collapse_nth(tree, 0);
            std::vector<Sym::Symbol> reversed(collapsed.size());
            const size_t new_size = collapsed.data()->compress_reverse_to(reversed.data());
            Sym::Symbol::copy_and_reverse_symbol_sequence(collapsed.data(), reversed.data(),
                                                          new_size);

            std::vector<Sym::Symbol> help_space(EXPRESSION_MAX_SYMBOL_COUNT);
            collapsed.data()->simplify(help_space.data());
            collapsed.resize(collapsed.data()->size());

            return collapsed;
        }
    }

    std::optional<std::vector<Symbol>> solve_integral(const std::vector<Symbol>& integral) {
        static constexpr size_t BLOCK_SIZE = 512;
        static constexpr size_t BLOCK_COUNT = 32;
        const size_t MAX_CHECK_COUNT =
            KnownIntegral::COUNT > Heuristic::COUNT ? KnownIntegral::COUNT : Heuristic::COUNT;
        const size_t SCAN_ARRAY_SIZE = MAX_CHECK_COUNT * MAX_EXPRESSION_COUNT;

        ExpressionArray<> expressions({single_integral_vacancy()}, EXPRESSION_MAX_SYMBOL_COUNT,
                                      MAX_EXPRESSION_COUNT);
        ExpressionArray<> expressions_swap(MAX_EXPRESSION_COUNT, EXPRESSION_MAX_SYMBOL_COUNT,
                                           expressions.size());

        ExpressionArray<SubexpressionCandidate> integrals({first_expression_candidate(integral)},
                                                          MAX_EXPRESSION_COUNT,
                                                          EXPRESSION_MAX_SYMBOL_COUNT);
        ExpressionArray<SubexpressionCandidate> integrals_swap(MAX_EXPRESSION_COUNT,
                                                               EXPRESSION_MAX_SYMBOL_COUNT);
        ExpressionArray<> help_spaces(MAX_EXPRESSION_COUNT, EXPRESSION_MAX_SYMBOL_COUNT,
                                      integrals.size());
        Util::DeviceArray<uint32_t> scan_array_1(SCAN_ARRAY_SIZE, true);
        Util::DeviceArray<uint32_t> scan_array_2(SCAN_ARRAY_SIZE, true);

        for (size_t i = 0;; ++i) {
            simplify<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, help_spaces);
            cudaDeviceSynchronize();

            check_for_known_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, scan_array_1);
            cudaDeviceSynchronize();

            thrust::inclusive_scan(thrust::device, scan_array_1.begin(), scan_array_1.end(),
                                   scan_array_1.data());
            cudaDeviceSynchronize();

            apply_known_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, expressions, help_spaces,
                                                               scan_array_1);
            cudaDeviceSynchronize();
            expressions.increment_size_from_device(scan_array_1.last());

            propagate_solved_subexpressions<<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions);
            cudaDeviceSynchronize();

            std::vector<Symbol> first_expression = expressions.to_vector(0);
            if (first_expression.data()->as<SubexpressionVacancy>().is_solved == 1) {
                return collapse(expressions.to_vector());
            }

            scan_array_1.zero_mem();
            find_redundand_expressions<<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions, scan_array_1);
            cudaDeviceSynchronize();

            scan_array_2.zero_mem(); // TODO: Not necessary?
            find_redundand_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, expressions,
                                                                  scan_array_1, scan_array_2);
            cudaDeviceSynchronize();

            thrust::inclusive_scan(thrust::device, scan_array_1.begin(), scan_array_1.end(),
                                   scan_array_1.data());
            thrust::inclusive_scan(thrust::device, scan_array_2.begin(), scan_array_2.end(),
                                   scan_array_2.data());
            cudaDeviceSynchronize();

            remove_expressions<true>
                <<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions, scan_array_1, expressions_swap);
            remove_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, scan_array_2, scan_array_1,
                                                          integrals_swap);
            cudaDeviceSynchronize();

            std::swap(expressions, expressions_swap);
            std::swap(integrals, integrals_swap);
            expressions.resize_from_device(scan_array_1.last());
            integrals.resize_from_device(scan_array_2.last());

            scan_array_1.zero_mem();
            scan_array_2.zero_mem();
            check_heuristics_applicability<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, expressions,
                                                                        scan_array_1, scan_array_2);
            cudaDeviceSynchronize();

            thrust::inclusive_scan(thrust::device, scan_array_1.begin(), scan_array_1.end(),
                                   scan_array_1.data());
            thrust::inclusive_scan(thrust::device, scan_array_2.begin(), scan_array_2.end(),
                                   scan_array_2.data());
            cudaDeviceSynchronize();

            apply_heuristics<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, integrals_swap, expressions,
                                                          help_spaces, scan_array_1, scan_array_2);
            cudaDeviceSynchronize();

            std::swap(integrals, integrals_swap);
            integrals.resize_from_device(scan_array_1.last());
            expressions.increment_size_from_device(scan_array_2.last());

            scan_array_1.set_mem(1);
            cudaDeviceSynchronize();

            propagate_failures_upwards<<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions, scan_array_1);
            cudaDeviceSynchronize();

            // First expression in the array has failed, all is lost
            if (scan_array_1.to_cpu(0) == 0) {
                return std::nullopt;
            }

            propagate_failures_downwards<<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions, scan_array_1);
            cudaDeviceSynchronize();

            scan_array_2.zero_mem();
            find_redundand_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, scan_array_1,
                                                                  scan_array_2);
            cudaDeviceSynchronize();

            thrust::inclusive_scan(thrust::device, scan_array_1.begin(), scan_array_1.end(),
                                   scan_array_1.data());
            thrust::inclusive_scan(thrust::device, scan_array_2.begin(), scan_array_2.end(),
                                   scan_array_2.data());
            cudaDeviceSynchronize();

            remove_expressions<false>
                <<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions, scan_array_1, expressions_swap);
            remove_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, scan_array_2, scan_array_1,
                                                          integrals_swap);
            cudaDeviceSynchronize();

            std::swap(expressions, expressions_swap);
            std::swap(integrals, integrals_swap);

            // How many expressions are left, cannot take scan_array_1.last() because space
            // after last place in scan_array_1 that corresponds to an expression is occupied
            // by ones
            expressions.resize(scan_array_1.to_cpu(expressions_swap.size() - 1));
            integrals.resize_from_device(scan_array_2.last());
            cudaDeviceSynchronize();

            scan_array_1.zero_mem();
            scan_array_2.zero_mem();
        }

        return std::nullopt;
    }
}
