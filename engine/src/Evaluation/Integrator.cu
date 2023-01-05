#include "Evaluation/Heuristic/Heuristic.cuh"
#include "Integrator.cuh"

#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "Collapser.cuh"
#include "IntegratorKernels.cuh"
#include "StaticFunctions.cuh"

#include "Symbol/SubexpressionCandidate.cuh"
#include "Utils/CompileConstants.cuh"
#include "Utils/Meta.cuh"

namespace Sym {
    namespace {
        struct EvaluationStatusChecker {
            __device__ bool operator()(EvaluationStatus status) const {
                return status == EvaluationStatus::Done;
            }
        };
    }

    Integrator::Integrator() :
        CHECK_COUNT(KnownIntegral::COUNT > Heuristic::COUNT ? KnownIntegral::COUNT
                                                            : Heuristic::COUNT),
        expressions(INITIAL_ARRAYS_SYMBOLS_CAPACITY, INITIAL_ARRAYS_EXPRESSIONS_CAPACITY),
        expressions_swap(INITIAL_ARRAYS_SYMBOLS_CAPACITY, INITIAL_ARRAYS_EXPRESSIONS_CAPACITY),
        integrals(INITIAL_ARRAYS_SYMBOLS_CAPACITY, INITIAL_ARRAYS_EXPRESSIONS_CAPACITY),
        integrals_swap(INITIAL_ARRAYS_SYMBOLS_CAPACITY, INITIAL_ARRAYS_EXPRESSIONS_CAPACITY),
        help_space(INITIAL_ARRAYS_SYMBOLS_CAPACITY * HELP_SPACE_MULTIPLIER * CHECK_COUNT,
                   INITIAL_ARRAYS_EXPRESSIONS_CAPACITY * CHECK_COUNT),
        scan_array_1(CHECK_COUNT * INITIAL_ARRAYS_EXPRESSIONS_CAPACITY),
        scan_array_2(CHECK_COUNT * INITIAL_ARRAYS_EXPRESSIONS_CAPACITY) {}

    void Integrator::reset_evaluation_statuses(
        Util::DeviceArray<EvaluationStatus>& evaluation_statuses) {
        evaluation_statuses.set_mem(EvaluationStatus::Incomplete);
    }

    void
    Integrator::resize_evaluation_statuses(Util::DeviceArray<EvaluationStatus>& evaluation_statuses,
                                           const size_t size) {
        if (evaluation_statuses.size() < size) {
            evaluation_statuses.resize(size * REALLOC_MULTIPLIER);
        }
    }

    bool Integrator::are_evaluation_statuses_done(
        const Util::DeviceArray<EvaluationStatus>& evaluation_statuses, const size_t count) {
        return thrust::count_if(thrust::device, evaluation_statuses.at(0),
                                evaluation_statuses.at(count), EvaluationStatusChecker()) == count;
    }

    void Integrator::resize_scan_arrays(const size_t size) {
        if (scan_array_1.size() < size) {
            scan_array_1.resize(size * REALLOC_MULTIPLIER);
            scan_array_2.resize(size * REALLOC_MULTIPLIER);
        }
    }

    void Integrator::simplify_integrals() {
        integrals_swap.reoffset_like<SubexpressionCandidate>(integrals.iterator());
        help_space.reoffset_like<SubexpressionCandidate>(integrals.iterator(),
                                                         HELP_SPACE_MULTIPLIER);
        resize_evaluation_statuses(evaluation_statuses_1, integrals.size());
        reset_evaluation_statuses(evaluation_statuses_1);

        const size_t block_count = Util::block_count(integrals.size(), BLOCK_SIZE);

        bool success = false;
        while (!success) {
            Kernel::simplify<<<block_count, BLOCK_SIZE>>>(integrals, integrals_swap, help_space,
                                                          evaluation_statuses_1);
            cudaDeviceSynchronize();

            success = are_evaluation_statuses_done(evaluation_statuses_1, integrals.size());
            if (!success) {
                integrals_swap.reoffset_indices(evaluation_statuses_1);
                help_space.reoffset_like<SubexpressionCandidate>(integrals_swap.iterator(),
                                                                 HELP_SPACE_MULTIPLIER);
            }
        }

        std::swap(integrals, integrals_swap);
    }

    void Integrator::check_for_known_integrals() {
        resize_scan_arrays(integrals.size() * CHECK_COUNT);
        scan_array_1.zero_mem();

        Kernel::check_for_known_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, scan_array_1);
        cudaDeviceSynchronize();

        thrust::inclusive_scan(thrust::device, scan_array_1.begin(), scan_array_1.end(),
                               scan_array_1.data());
        cudaDeviceSynchronize();
    }

    void Integrator::apply_known_integrals() {
        const size_t expression_count_diff = scan_array_1.last_cpu();
        resize_evaluation_statuses(evaluation_statuses_1, expression_count_diff);
        reset_evaluation_statuses(evaluation_statuses_1);

        const size_t old_expression_count = expressions.size();
        const size_t new_expression_count = old_expression_count + expression_count_diff;

        // No applications possible
        if (old_expression_count == new_expression_count) {
            return;
        }

        expressions.resize(new_expression_count, INITIAL_EXPRESSIONS_CAPACITY);

        const auto new_expressions = expressions.iterator(old_expression_count);
        help_space.reoffset_like(new_expressions, HELP_SPACE_MULTIPLIER);

        bool success = false;
        while (!success) {
            Kernel::apply_known_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(
                integrals, expressions, new_expressions.index(), help_space, scan_array_1,
                evaluation_statuses_1);
            cudaDeviceSynchronize();

            success = are_evaluation_statuses_done(evaluation_statuses_1, expression_count_diff);
            if (!success) {
                expressions.reoffset_indices(evaluation_statuses_1, old_expression_count);
                help_space.reoffset_like(new_expressions, HELP_SPACE_MULTIPLIER);
            }
        }

        Kernel::propagate_solved_subexpressions<<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions);
        cudaDeviceSynchronize();
    }

    bool Integrator::is_original_expression_solved() {
        std::vector<Symbol> first_expression = expressions.to_vector(0);
        return first_expression.data()->as<SubexpressionVacancy>().is_solved == 1;
    }

    void Integrator::remove_unnecessary_candidates() {
        resize_scan_arrays(std::max(expressions.size(), integrals.size()));
        scan_array_1.zero_mem();
        scan_array_2.zero_mem();
        cudaDeviceSynchronize();

        Kernel::find_redundand_expressions<<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions, scan_array_1);
        cudaDeviceSynchronize();

        Kernel::find_redundand_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, expressions,
                                                                      scan_array_1, scan_array_2);
        cudaDeviceSynchronize();

        thrust::inclusive_scan(thrust::device, scan_array_1.begin(), scan_array_1.end(),
                               scan_array_1.data());
        thrust::inclusive_scan(thrust::device, scan_array_2.begin(), scan_array_2.end(),
                               scan_array_2.data());
        cudaDeviceSynchronize();

        const size_t new_expression_count = scan_array_1.last_cpu();
        const size_t new_integral_count = scan_array_2.last_cpu();

        expressions_swap.reoffset_like_scan(expressions, scan_array_1);
        integrals_swap.reoffset_like_scan<SubexpressionCandidate>(integrals, scan_array_2);

        Kernel::remove_expressions<true>
            <<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions, scan_array_1, expressions_swap);
        Kernel::remove_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, scan_array_2, scan_array_1,
                                                              integrals_swap);
        cudaDeviceSynchronize();

        std::swap(expressions, expressions_swap);
        std::swap(integrals, integrals_swap);
    }

    void Integrator::check_heuristics_applicability() {
        resize_scan_arrays(integrals.size() * CHECK_COUNT);
        scan_array_1.zero_mem();
        scan_array_2.zero_mem();
        cudaDeviceSynchronize();

        help_space.reoffset_like<SubexpressionCandidate>(integrals.iterator(), HELP_SPACE_MULTIPLIER, Heuristic::COUNT);

        Kernel::check_heuristics_applicability<<<BLOCK_COUNT, BLOCK_SIZE>>>(
            integrals, expressions, help_space, scan_array_1, scan_array_2);
        cudaDeviceSynchronize();

        thrust::inclusive_scan(thrust::device, scan_array_1.begin(), scan_array_1.end(),
                               scan_array_1.data());
        thrust::inclusive_scan(thrust::device, scan_array_2.begin(), scan_array_2.end(),
                               scan_array_2.data());
        cudaDeviceSynchronize();
    }

    void Integrator::apply_heuristics() {
        const size_t new_integral_count = scan_array_1.last_cpu();

        const size_t old_expression_count = expressions.size();
        const size_t expression_count_diff = scan_array_2.last_cpu();
        const size_t new_expression_count = old_expression_count + expression_count_diff;

        // No applications possible
        if (new_integral_count == 0) {
            return;
        }

        resize_evaluation_statuses(evaluation_statuses_1, new_integral_count);
        reset_evaluation_statuses(evaluation_statuses_1);
        resize_evaluation_statuses(evaluation_statuses_2, expression_count_diff);
        reset_evaluation_statuses(evaluation_statuses_2);

        expressions.resize(new_expression_count, INITIAL_EXPRESSIONS_CAPACITY);
        integrals_swap.resize(new_integral_count, INITIAL_EXPRESSIONS_CAPACITY);

        // If old_expression_count == new_expression_count, then there wont be any new expressions,
        // and so we don't need to allocate space for them. We do however still have to create a
        // valid iterator, so we just create an iterator to the first expression in this case
        const size_t new_expressions_offset =
            old_expression_count == new_expression_count ? 0 : old_expression_count;

        const auto new_integrals = integrals_swap.iterator();
        const auto new_expressions = expressions.iterator(new_expressions_offset);

        help_space.reoffset_like<SubexpressionCandidate>(new_integrals, HELP_SPACE_MULTIPLIER);

        bool success = false;
        while (!success) {
            Kernel::apply_heuristics<<<BLOCK_COUNT, BLOCK_SIZE>>>(
                integrals, integrals_swap, expressions, new_expressions_offset, help_space,
                scan_array_1, scan_array_2, evaluation_statuses_1, evaluation_statuses_2);
            cudaDeviceSynchronize();

            // No need to check evaluation_statuses_2 as a reallocation request will appear there
            // only if one appears in evaluation_statuses_1
            success = are_evaluation_statuses_done(evaluation_statuses_1, new_integral_count);
            if (!success) {
                integrals_swap.reoffset_indices(evaluation_statuses_1);
                expressions_swap.reoffset_indices(evaluation_statuses_2, old_expression_count);
                help_space.reoffset_like<SubexpressionCandidate>(integrals_swap.iterator(),
                                                                 HELP_SPACE_MULTIPLIER);
            }
        }

        std::swap(integrals, integrals_swap);

        scan_array_1.set_mem(1);
        Kernel::propagate_failures_upwards<<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions, scan_array_1);
        cudaDeviceSynchronize();
    }

    bool Integrator::has_original_expression_failed() { return scan_array_1.to_cpu(0) == 0; }

    void Integrator::remove_failed_candidates() {
        Kernel::propagate_failures_downwards<<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions,
                                                                          scan_array_1);
        cudaDeviceSynchronize();

        resize_scan_arrays(std::max(integrals.size(), expressions.size()));
        scan_array_2.zero_mem();
        Kernel::find_redundand_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, scan_array_1,
                                                                      scan_array_2);
        cudaDeviceSynchronize();

        thrust::inclusive_scan(thrust::device, scan_array_1.begin(), scan_array_1.end(),
                               scan_array_1.data());
        thrust::inclusive_scan(thrust::device, scan_array_2.begin(), scan_array_2.end(),
                               scan_array_2.data());
        cudaDeviceSynchronize();

        // How many expressions are left, cannot take scan_array_1.last() because space
        // after last place in scan_array_1 that corresponds to an expression is occupied
        // by ones
        const size_t new_expression_count = scan_array_1.to_cpu(expressions_swap.size() - 1);
        const size_t new_integral_count = scan_array_2.last_cpu();

        expressions_swap.reoffset_like_scan(expressions, scan_array_1);
        integrals_swap.reoffset_like_scan<SubexpressionCandidate>(integrals, scan_array_2);

        Kernel::remove_expressions<false>
            <<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions, scan_array_1, expressions_swap);

        Kernel::remove_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, scan_array_2, scan_array_1,
                                                              integrals_swap);
        cudaDeviceSynchronize();

        std::swap(expressions, expressions_swap);
        std::swap(integrals, integrals_swap);

        scan_array_1.zero_mem();
        scan_array_2.zero_mem();
        cudaDeviceSynchronize();
    }

    std::optional<std::vector<Symbol>>
    Integrator::solve_integral(const std::vector<Symbol>& integral) {
        expressions.load_from_vector({single_integral_vacancy()});
        integrals.load_from_vector({first_expression_candidate(integral)});

        for (size_t i = 0;; ++i) {
            simplify_integrals();

            printf("SIMPL:\n%s\n\n", integrals.to_string().c_str());

            check_for_known_integrals();
            apply_known_integrals();

            if (is_original_expression_solved()) {
                return Collapser::collapse(expressions.to_vector());
            }

            remove_unnecessary_candidates();

            check_heuristics_applicability();
            apply_heuristics();

            printf("HEUR:\n%s\n\n", integrals.to_string().c_str());

            if (has_original_expression_failed()) {
                return std::nullopt;
            }

            remove_failed_candidates();
        }

        return std::nullopt;
    }
}
