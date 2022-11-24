#include "IntegratorKernels.cuh"

#include "Heuristic/Heuristic.cuh"
#include "KnownIntegral/KnownIntegral.cuh"

namespace Sym {
    __device__ bool is_nonzero(const size_t index,
                               const Util::DeviceArray<uint32_t>& inclusive_scan) {
        return index == 0 && inclusive_scan[index] != 0 ||
               index != 0 && inclusive_scan[index - 1] != inclusive_scan[index];
    }
}

namespace Sym::Kernel {
    namespace {
        constexpr size_t TRANSFORM_GROUP_SIZE = 32;
    }

    /*
     * @brief Try to set `expressions[potential_solver_idx]` (SubexpressionCandidate)
     * as a solution to its SubexpressionVacancy
     *
     * @param expressions Expressions array with a candidate to solve and a missing
     * subexpression
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
            &expressions[vacancy_expr_idx].subexpression_candidate.subexpressions_left, 1) - 1;

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

    __global__ void simplify(const ExpressionArray<> expressions, ExpressionArray<> destination,
                             ExpressionArray<> help_spaces) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        for (size_t expr_idx = thread_idx; expr_idx < expressions.size();
             expr_idx += thread_count) {
            expressions[expr_idx].copy_to(&destination[expr_idx]);
            destination[expr_idx].simplify(help_spaces.at(expr_idx));
        }
    }

    __global__ void
    check_for_known_integrals(const ExpressionArray<SubexpressionCandidate> integrals,
                              Util::DeviceArray<uint32_t> applicability) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        const size_t check_step = thread_count / TRANSFORM_GROUP_SIZE;

        for (size_t check_idx = thread_idx / TRANSFORM_GROUP_SIZE; check_idx < KnownIntegral::COUNT;
             check_idx += check_step) {
            for (size_t int_idx = thread_idx % TRANSFORM_GROUP_SIZE; int_idx < integrals.size();
                 int_idx += TRANSFORM_GROUP_SIZE) {
                size_t appl_idx = MAX_EXPRESSION_COUNT * check_idx + int_idx;
                applicability[appl_idx] =
                    KnownIntegral::CHECKS[check_idx](integrals[int_idx].arg().as<Integral>());
            }
        }
    }

    __global__ void apply_known_integrals(const ExpressionArray<SubexpressionCandidate> integrals,
                                          ExpressionArray<> expressions,
                                          ExpressionArray<> help_spaces,
                                          const Util::DeviceArray<uint32_t> applicability) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        const size_t trans_step = thread_count / TRANSFORM_GROUP_SIZE;

        for (size_t trans_idx = thread_idx / TRANSFORM_GROUP_SIZE; trans_idx < KnownIntegral::COUNT;
             trans_idx += trans_step) {
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

    __global__ void propagate_solved_subexpressions(ExpressionArray<> expressions) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        // For each tree node there is a seperate starting thread.
        // If its node is solved it moves to it's parent.
        // It tries to fill the parent's vacancy with it's own solution.
        // If it succeeds and all of the parent's vacancies are solved, it moves to the parent.
        // This operation upwards is repeated upwards while all solutions to the current node
        // exists and the parent's vacancy is not solved and ends at the root.

        // Since `expr_idx = 0` is SubexpressionVacancy of the original integral, it is skipped
        for (size_t expr_idx = thread_idx + 1; expr_idx < expressions.size();
             expr_idx += thread_count) {
            size_t current_expr_idx = expr_idx;
            while (current_expr_idx != 0) {
                printf("%lu\n",current_expr_idx);
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
                current_expr_idx =
                    expressions[current_expr_idx].subexpression_candidate.vacancy_expression_idx;
            }
        }
    }

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
                const size_t& parent_idx =
                    expressions[current_expr_idx].subexpression_candidate.vacancy_expression_idx;
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

    __global__ void
    find_redundand_integrals(const ExpressionArray<> integrals, const ExpressionArray<> expressions,
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

            integrals_removability[int_idx] = parent_expr_failed || parent_vacancy_solved ? 0 : 1;
        }
    }

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
                    atomicAdd(&parent_vacancy.candidate_expression_count, result.new_expressions);
                }
            }
        }
    }

    __global__ void apply_heuristics(const ExpressionArray<SubexpressionCandidate> integrals,
                                     ExpressionArray<> integrals_destinations,
                                     ExpressionArray<> expressions_destinations,
                                     ExpressionArray<> help_spaces,
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
                    const size_t expr_dst_idx = expressions_destinations.size() +
                                                index_from_scan(new_expressions_indices, appl_idx);
                    Heuristic::APPLICATIONS[trans_idx](
                        integrals[int_idx], integrals_destinations.iterator(int_dst_idx),
                        expressions_destinations.iterator(expr_dst_idx), help_spaces[int_dst_idx]);
                }
                else {
                    Heuristic::APPLICATIONS[trans_idx](
                        integrals[int_idx], integrals_destinations.iterator(int_dst_idx),
                        ExpressionArray<>::Iterator::null(), help_spaces[int_dst_idx]);
                }
            }
        }
    }

    __global__ void propagate_failures_upwards(ExpressionArray<> expressions,
                                               Util::DeviceArray<uint32_t> failures) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        for (size_t expr_idx = thread_idx; expr_idx < expressions.size();
             expr_idx += thread_count) {
            SubexpressionCandidate& self_candidate = expressions[expr_idx].subexpression_candidate;

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
                const size_t& parent_idx =
                    expressions[current_expr_idx].subexpression_candidate.vacancy_expression_idx;
                const size_t& vacancy_idx =
                    expressions[current_expr_idx].subexpression_candidate.vacancy_idx;
                SubexpressionVacancy& parent_vacancy =
                    expressions[parent_idx][vacancy_idx].subexpression_vacancy;

                if (parent_vacancy.candidate_integral_count != 0 || parent_vacancy.is_solved == 1) {
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

    __global__ void propagate_failures_downwards(ExpressionArray<> expressions,
                                                 Util::DeviceArray<uint32_t> failures) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        // Top expression has no parents, so we skip it
        for (size_t expr_idx = thread_idx + 1; expr_idx < expressions.size();
             expr_idx += thread_count) {
            size_t current_expr_idx = expr_idx;

            while (current_expr_idx != 0) {
                const size_t& parent_idx =
                    expressions[current_expr_idx].subexpression_candidate.vacancy_expression_idx;

                if (failures[parent_idx] == 0) {
                    failures[expr_idx] = 0;
                    break;
                }

                current_expr_idx = parent_idx;
            }
        }
    }

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

}
