#include "Integrator.cuh"

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "IntegratorKernels.cuh"
#include "StaticFunctions.cuh"

#include "Utils/CompileConstants.cuh"
#include "Utils/Meta.cuh"

namespace Sym {
    std::vector<Sym::Symbol>
    Integrator::replace_nth_with_tree(std::vector<Sym::Symbol> expression, const size_t n,
                                      const std::vector<Sym::Symbol>& tree) {
        if constexpr (Consts::DEBUG) {
            if (!tree[0].is(Type::SubexpressionCandidate)) {
                Util::crash("Invalid first symbol of tree: %s, should be SubexpressionCandidate",
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

    std::vector<Sym::Symbol>
    Integrator::collapse_nth(const std::vector<std::vector<Sym::Symbol>>& tree, const size_t n) {
        std::vector<Sym::Symbol> current_collapse = tree[n];

        for (size_t i = 0; i < current_collapse.size(); ++i) {
            if (!current_collapse[i].is(Sym::Type::SubexpressionVacancy)) {
                continue;
            }

            const auto subtree =
                collapse_nth(tree, current_collapse[i].as<Sym::SubexpressionVacancy>().solver_idx);

            auto new_collapse = replace_nth_with_tree(current_collapse, i, subtree);
            i += new_collapse.size() - current_collapse.size();
            current_collapse = new_collapse;
        }

        return current_collapse;
    }

    std::vector<Sym::Symbol>
    Integrator::collapse(const std::vector<std::vector<Sym::Symbol>>& tree) {
        auto collapsed = collapse_nth(tree, 0);
        std::vector<Sym::Symbol> reversed(collapsed.size());
        const size_t new_size = collapsed.data()->compress_reverse_to(reversed.data());
        Sym::Symbol::copy_and_reverse_symbol_sequence(collapsed.data(), reversed.data(), new_size);

        std::vector<Sym::Symbol> help_space(EXPRESSION_MAX_SYMBOL_COUNT);
        collapsed.data()->simplify(help_space.data());
        collapsed.resize(collapsed.data()->size());

        return collapsed;
    }

    Integrator::Integrator() :
        MAX_CHECK_COUNT(KnownIntegral::COUNT > Heuristic::COUNT ? KnownIntegral::COUNT
                                                                : Heuristic::COUNT),
        SCAN_ARRAY_SIZE(MAX_CHECK_COUNT * MAX_EXPRESSION_COUNT),
        expressions(MAX_EXPRESSION_COUNT, EXPRESSION_MAX_SYMBOL_COUNT),
        expressions_swap(MAX_EXPRESSION_COUNT, EXPRESSION_MAX_SYMBOL_COUNT),
        integrals(MAX_EXPRESSION_COUNT, EXPRESSION_MAX_SYMBOL_COUNT),
        integrals_swap(MAX_EXPRESSION_COUNT, EXPRESSION_MAX_SYMBOL_COUNT),
        help_space(MAX_EXPRESSION_COUNT, EXPRESSION_MAX_SYMBOL_COUNT, integrals.size()),
        scan_array_1(SCAN_ARRAY_SIZE, true),
        scan_array_2(SCAN_ARRAY_SIZE, true) {}

    void Integrator::simplify_integrals() {
        integrals_swap.resize(integrals.size());
        Kernel::simplify<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, integrals_swap, help_space);
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
                                                                   help_space, scan_array_1);
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
            integrals, expressions, scan_array_1, scan_array_2);
        cudaDeviceSynchronize();

        thrust::inclusive_scan(thrust::device, scan_array_1.begin(), scan_array_1.end(),
                               scan_array_1.data());
        thrust::inclusive_scan(thrust::device, scan_array_2.begin(), scan_array_2.end(),
                               scan_array_2.data());
        cudaDeviceSynchronize();
    }

    void Integrator::apply_heuristics() {
        Kernel::apply_heuristics<<<BLOCK_COUNT, BLOCK_SIZE>>>(
            integrals, integrals_swap, expressions, help_space, scan_array_1, scan_array_2);
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
                return collapse(expressions.to_vector());
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
