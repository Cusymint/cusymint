#include "Integrate.cuh"

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "Heuristic/Heuristic.cuh"
#include "IntegrateKernels.cuh"
#include "KnownIntegral/KnownIntegral.cuh"
#include "StaticFunctions.cuh"

#include "Utils/CompileConstants.cuh"
#include "Utils/Meta.cuh"

namespace Sym {
    namespace {
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
            integrals_swap.resize(integrals.size());
            Kernel::simplify<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, integrals_swap, help_spaces);
            cudaDeviceSynchronize();
            std::swap(integrals, integrals_swap);

            Kernel::check_for_known_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, scan_array_1);
            cudaDeviceSynchronize();

            thrust::inclusive_scan(thrust::device, scan_array_1.begin(), scan_array_1.end(),
                                   scan_array_1.data());
            cudaDeviceSynchronize();

            Kernel::apply_known_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, expressions,
                                                                       help_spaces, scan_array_1);
            cudaDeviceSynchronize();
            expressions.increment_size_from_device(scan_array_1.last());

            Kernel::propagate_solved_subexpressions<<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions);
            cudaDeviceSynchronize();

            std::vector<Symbol> first_expression = expressions.to_vector(0);
            if (first_expression.data()->as<SubexpressionVacancy>().is_solved == 1) {
                return collapse(expressions.to_vector());
            }

            scan_array_1.zero_mem();
            Kernel::find_redundand_expressions<<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions,
                                                                            scan_array_1);
            cudaDeviceSynchronize();

            Kernel::find_redundand_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(
                integrals, expressions, scan_array_1, scan_array_2);
            cudaDeviceSynchronize();

            thrust::inclusive_scan(thrust::device, scan_array_1.begin(), scan_array_1.end(),
                                   scan_array_1.data());
            thrust::inclusive_scan(thrust::device, scan_array_2.begin(), scan_array_2.end(),
                                   scan_array_2.data());
            cudaDeviceSynchronize();

            Kernel::remove_expressions<true>
                <<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions, scan_array_1, expressions_swap);
            Kernel::remove_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, scan_array_2,
                                                                  scan_array_1, integrals_swap);
            cudaDeviceSynchronize();

            std::swap(expressions, expressions_swap);
            std::swap(integrals, integrals_swap);
            expressions.resize_from_device(scan_array_1.last());
            integrals.resize_from_device(scan_array_2.last());

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

            Kernel::apply_heuristics<<<BLOCK_COUNT, BLOCK_SIZE>>>(
                integrals, integrals_swap, expressions, help_spaces, scan_array_1, scan_array_2);
            cudaDeviceSynchronize();

            std::swap(integrals, integrals_swap);
            integrals.resize_from_device(scan_array_1.last());
            expressions.increment_size_from_device(scan_array_2.last());

            scan_array_1.set_mem(1);
            cudaDeviceSynchronize();

            Kernel::propagate_failures_upwards<<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions,
                                                                            scan_array_1);
            cudaDeviceSynchronize();

            // First expression in the array has failed, all is lost
            if (scan_array_1.to_cpu(0) == 0) {
                return std::nullopt;
            }

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
            Kernel::remove_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, scan_array_2,
                                                                  scan_array_1, integrals_swap);
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
