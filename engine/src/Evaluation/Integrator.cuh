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
