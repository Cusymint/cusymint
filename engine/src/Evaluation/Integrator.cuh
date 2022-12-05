#ifndef INTEGRATE_CUH
#define INTEGRATE_CUH

#include <optional>

#include "Symbol/ExpressionArray.cuh"
#include "Symbol/Symbol.cuh"

#include "Heuristic/Heuristic.cuh"
#include "KnownIntegral/KnownIntegral.cuh"
#include "Status.cuh"

namespace Sym {
    class Integrator {
        // How many symbols can an expression hold initially
        static constexpr size_t INITIAL_EXPRESSIONS_CAPACITY = 128;
        // How many expressions of size `INITIAL_EXPRESSION_CAPACITIES` can an array hold initially
        static constexpr size_t INITIAL_ARRAYS_EXPRESSIONS_CAPACITY = 64;
        static constexpr size_t INITIAL_ARRAYS_SYMBOLS_CAPACITY =
            INITIAL_ARRAYS_EXPRESSIONS_CAPACITY * INITIAL_EXPRESSIONS_CAPACITY;

        // Sizes of `scan_array_X` and `evaluation_statuses` are multiplied by this value on
        // reallocation
        static constexpr size_t REALLOC_MULTIPLIER = 2;
        static constexpr size_t HELP_SPACE_MULTIPLIER = 2;

        const size_t CHECK_COUNT;

        ExpressionArray<> expressions;
        ExpressionArray<> expressions_swap;

        ExpressionArray<SubexpressionCandidate> integrals;
        ExpressionArray<SubexpressionCandidate> integrals_swap;

        ExpressionArray<> help_space;

        // Scan arrays used in various algorithms. Their size can be larger than the actually used
        // part.
        Util::DeviceArray<uint32_t> scan_array_1;
        Util::DeviceArray<uint32_t> scan_array_2;

        // EvaluationStatus arrays used for checking reallocation requests. Their size can be larger
        // than the actually used part.
        Util::DeviceArray<EvaluationStatus> evaluation_statuses_1;
        Util::DeviceArray<EvaluationStatus> evaluation_statuses_2;

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
         * @brief Sets all evaluation statuses to `EvaluationStatus::Incomplete`
         */
        static void
        reset_evaluation_statuses(Util::DeviceArray<EvaluationStatus>& evaluation_statuses);

        /*
         * @brief Resizes `evaluation_statuses` to at least `size` if its size is smaller than
         * `size`, does nothing if its size is already at least `size`
         */
        static void
        resize_evaluation_statuses(Util::DeviceArray<EvaluationStatus>& evaluation_statuses,
                                   const size_t size);

        /*
         * @brief Checks if the first `count` statuses are equal to `EvaluationStatus::Done`
         */
        static bool
        are_evaluation_statuses_done(const Util::DeviceArray<EvaluationStatus>& evaluation_statuses,
                                     const size_t count);

        /*
         * @brief Resizes `scan_array_1` and `scan_array_2` to at least `size` if their size is
         * smaller than `size`, does nothing if their size is already at least `size`
         */
        void resize_scan_arrays(const size_t size);

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
