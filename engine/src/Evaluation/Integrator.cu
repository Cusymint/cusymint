#include "Evaluation/Heuristic/Heuristic.cuh"
#include "Integrator.cuh"

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "Collapser.cuh"
#include "IntegratorKernels.cuh"
#include "StaticFunctions.cuh"

#include "Utils/CompileConstants.cuh"
#include "Utils/Meta.cuh"

namespace Sym {
    Integrator::Integrator() :
        MAX_CHECK_COUNT(KnownIntegral::COUNT > Heuristic::COUNT ? KnownIntegral::COUNT
                                                                : Heuristic::COUNT),
        SCAN_ARRAY_SIZE(MAX_CHECK_COUNT * MAX_EXPRESSION_COUNT),
        expressions(MAX_EXPRESSION_COUNT, EXPRESSION_MAX_SYMBOL_COUNT),
        expressions_swap(MAX_EXPRESSION_COUNT, EXPRESSION_MAX_SYMBOL_COUNT),
        integrals(MAX_EXPRESSION_COUNT, EXPRESSION_MAX_SYMBOL_COUNT),
        integrals_swap(MAX_EXPRESSION_COUNT, EXPRESSION_MAX_SYMBOL_COUNT),
        help_spaces(MAX_EXPRESSION_COUNT * Heuristic::COUNT, EXPRESSION_MAX_SYMBOL_COUNT,
                    integrals.size()),
        scan_array_1(SCAN_ARRAY_SIZE, true),
        scan_array_2(SCAN_ARRAY_SIZE, true) {}

    void Integrator::simplify_integrals() {
        integrals_swap.resize(integrals.size());
        Kernel::simplify<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, integrals_swap, help_spaces);
        cudaDeviceSynchronize();
        std::swap(integrals, integrals_swap);
    }

    void Integrator::check_for_known_integrals() {
        Kernel::check_for_known_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, scan_array_1);
        cudaDeviceSynchronize();

        thrust::inclusive_scan(thrust::device, scan_array_1.begin(), scan_array_1.end(),
                               scan_array_1.data());
        cudaDeviceSynchronize();
    }

    void Integrator::apply_known_integrals() {
        Kernel::apply_known_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, expressions,
                                                                   help_spaces, scan_array_1);
        cudaDeviceSynchronize();
        expressions.increment_size_from_device(scan_array_1.last());

        Kernel::propagate_solved_subexpressions<<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions);
        cudaDeviceSynchronize();
    }

    bool Integrator::is_original_expression_solved() {
        std::vector<Symbol> first_expression = expressions.to_vector(0);
        return first_expression.data()->as<SubexpressionVacancy>().is_solved == 1;
    }

    void Integrator::remove_unnecessary_candidates() {
        scan_array_1.zero_mem();
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

        Kernel::remove_expressions<true>
            <<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions, scan_array_1, expressions_swap);
        Kernel::remove_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, scan_array_2, scan_array_1,
                                                              integrals_swap);
        cudaDeviceSynchronize();

        std::swap(expressions, expressions_swap);
        std::swap(integrals, integrals_swap);
        expressions.resize_from_device(scan_array_1.last());
        integrals.resize_from_device(scan_array_2.last());
    }

    void Integrator::check_heuristics_applicability() {
        scan_array_1.zero_mem();
        scan_array_2.zero_mem();
        Kernel::check_heuristics_applicability<<<BLOCK_COUNT, BLOCK_SIZE>>>(
            integrals, expressions, help_spaces, scan_array_1, scan_array_2);
        cudaDeviceSynchronize();

        thrust::inclusive_scan(thrust::device, scan_array_1.begin(), scan_array_1.end(),
                               scan_array_1.data());
        thrust::inclusive_scan(thrust::device, scan_array_2.begin(), scan_array_2.end(),
                               scan_array_2.data());
        cudaDeviceSynchronize();
    }

    void Integrator::apply_heuristics() {
        Kernel::apply_heuristics<<<BLOCK_COUNT, BLOCK_SIZE>>>(
            integrals, integrals_swap, expressions, help_spaces, scan_array_1, scan_array_2);
        cudaDeviceSynchronize();

        std::swap(integrals, integrals_swap);
        integrals.resize_from_device(scan_array_1.last());
        expressions.increment_size_from_device(scan_array_2.last());

        scan_array_1.set_mem(1);
        Kernel::propagate_failures_upwards<<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions, scan_array_1);
        cudaDeviceSynchronize();
    }

    bool Integrator::has_original_expression_failed() { return scan_array_1.to_cpu(0) == 0; }

    void Integrator::remove_failed_candidates() {
        Kernel::propagate_failures_downwards<<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions,
                                                                          scan_array_1);
        cudaDeviceSynchronize();

        scan_array_2.zero_mem();
        Kernel::find_redundand_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, scan_array_1,
                                                                      scan_array_2);
        cudaDeviceSynchronize();

        thrust::inclusive_scan(thrust::device, scan_array_1.begin(), scan_array_1.end(),
                               scan_array_1.data());
        thrust::inclusive_scan(thrust::device, scan_array_2.begin(), scan_array_2.end(),
                               scan_array_2.data());
        cudaDeviceSynchronize();

        Kernel::remove_expressions<false>
            <<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions, scan_array_1, expressions_swap);
        Kernel::remove_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, scan_array_2, scan_array_1,
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

    std::optional<std::vector<Symbol>>
    Integrator::solve_integral(const std::vector<Symbol>& integral) {
        expressions.load_from_vector({single_integral_vacancy()});
        integrals.load_from_vector({first_expression_candidate(integral)});

        for (size_t i = 0;; ++i) {
            simplify_integrals();

            check_for_known_integrals();
            apply_known_integrals();

            if (is_original_expression_solved()) {
                return Collapser::collapse(expressions.to_vector());
            }

            remove_unnecessary_candidates();

            check_heuristics_applicability();
            apply_heuristics();

            if (has_original_expression_failed()) {
                return std::nullopt;
            }

            remove_failed_candidates();
        }

        return std::nullopt;
    }
}
