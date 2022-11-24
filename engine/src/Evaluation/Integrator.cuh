#ifndef INTEGRATE_CUH
#define INTEGRATE_CUH

#include <optional>

#include "Symbol/ExpressionArray.cuh"
#include "Symbol/Symbol.cuh"

#include "Heuristic/Heuristic.cuh"
#include "KnownIntegral/KnownIntegral.cuh"

namespace Sym {
    /*
     * @brief Maximum number of symbols in a single expression
     */
    static constexpr size_t EXPRESSION_MAX_SYMBOL_COUNT = 512;

    class Integrator {
        const size_t MAX_CHECK_COUNT;
        const size_t SCAN_ARRAY_SIZE;

        ExpressionArray<> expressions;
        ExpressionArray<> expressions_swap;

        ExpressionArray<SubexpressionCandidate> integrals;
        ExpressionArray<SubexpressionCandidate> integrals_swap;

        ExpressionArray<> help_spaces;

        Util::DeviceArray<uint32_t> scan_array_1;
        Util::DeviceArray<uint32_t> scan_array_2;

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
                                                       const std::vector<Sym::Symbol>& tree);

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
                                              const size_t n);

        /*
         * @brief Collapses a tree of expressions with Solutions with Substitutions and
         * interreferencing SubexpressionCandidates and SubexpressionVacancies to a single
         * expression
         *
         * @param tree Tree to collapse
         *
         * @return Collapsed tree
         */
        std::vector<Sym::Symbol> collapse(const std::vector<std::vector<Sym::Symbol>>& tree);

        /*
         * @brief Simplifies integrals `integrals`
         */
        void simplify_integrals();

        /*
         * @brief Fills in `scan_array_1` with known integrals information
         */
        void check_for_known_integrals();

        /*
         * @brief Applies known integrals specified by `scan_array_1` and propagates successes
         */
        void apply_known_integrals();

        /*
         * @brief Checks if the first expression in `expressions` is solved
         */
        bool is_original_expression_solved();

        /*
         * @brief Removes candidates to already solved expressions.
         */
        void remove_unnecessary_candidates();

        /*
         * @brief Checks applicability of heuristics. Saves results for integrals in `scan_array_1`
         * and in `scan_array_2` for expressions.
         */
        void check_heuristics_applicability();

        /*
         * @brief Applies heuristics using information in `scan_array_1` and `scan_array_2`.
         * Propagates failures upwards.
         */
        void apply_heuristics();

        /*
         * @brief Checks if the first expression in `expressions` is failed. Uses information put
         * into `scan_array_1` by apply_heuristics.
         */
        bool has_original_expression_failed();

        /*
         * @brief Removes candidates to failed expressions.
         */
        void remove_failed_candidates();

      public:
        /*
         * @brief Block size of CUDA kernels used by `solve_integral`
         */
        static constexpr size_t BLOCK_SIZE = 512;

        /*
         * @brief Block count of CUDA kernels used by `solve_integral`
         */
        static constexpr size_t BLOCK_COUNT = 32;

        Integrator();

        /*
         * @brief Solves an integral and returns the result
         *
         * @param integral Vector of symbols with the integral, the first symbol should be
         * `Sym::Integral`
         *
         * @return `std::nullopt` if no result has been found, vector of vectors with the solution
         * tree otherwise
         */
        std::optional<std::vector<Sym::Symbol>>
        solve_integral(const std::vector<Sym::Symbol>& integral);
    };
}

#endif