#ifndef INTEGRATOR_CUH
#define INTEGRATOR_CUH

#include <optional>
#include <type_traits>

#include "Evaluation/Collapser.cuh"
#include "Symbol/ExpressionArray.cuh"
#include "Symbol/MetaOperators.cuh"
#include "Symbol/Symbol.cuh"

#include "Heuristic/Heuristic.cuh"
#include "KnownIntegral/KnownIntegral.cuh"
#include "Status.cuh"

#include "ComputationHistory.cuh"

namespace Sym {
    class Integrator {
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

        // Count of SubexpressionCandidate expressions ever created
        size_t candidates_created = 1;

        // EvaluationStatus arrays used for checking reallocation requests. Their size can be larger
        // than the actually used part.
        Util::DeviceArray<EvaluationStatus> evaluation_statuses_1;
        Util::DeviceArray<EvaluationStatus> evaluation_statuses_2;

        /*
         * @brief Increments `candidates_created` by value read from GPU
         */
        void increment_counter_from_device();

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
         * @brief How many times larger a help space expression is than the expression it
         * corresponds to
         */
        static constexpr size_t HELP_SPACE_MULTIPLIER = 2;

        /*
         * @brief Sizes of `scan_array_X` and `evaluation_statuses_X` are multiplied by this value
         * on reallocation
         */
        static constexpr size_t REALLOC_MULTIPLIER = 2;

        /*
         * @brief Block size of CUDA kernels used by `solve_integral`
         */
        static constexpr size_t BLOCK_SIZE = 1;

        /*
         * @brief Block count of CUDA kernels used by `solve_integral`
         */
        static constexpr size_t BLOCK_COUNT = 128;

        /*
         * @brief How many symbols can an expression hold initially
         */
        static constexpr size_t INITIAL_EXPRESSIONS_CAPACITY = 128;

        /*
         * @brief How many expressions of size `INITIAL_EXPRESSION_CAPACITIES` can an array hold
         * initially
         */
        static constexpr size_t INITIAL_ARRAYS_EXPRESSIONS_CAPACITY = 64;

        /*
         * @brief How many symbols should new arrays contain
         */
        static constexpr size_t INITIAL_ARRAYS_SYMBOLS_CAPACITY =
            INITIAL_ARRAYS_EXPRESSIONS_CAPACITY * INITIAL_EXPRESSIONS_CAPACITY;

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
        std::optional<std::vector<Symbol>> solve_integral(const std::vector<Symbol>& integral);

        /*
         * @brief Solves an integral, optionally saves computation history and returns the result
         *
         * @tparam WITH_HISTORY If `true` then computation steps will be dumped to `history`.
         * otherwise not. Parameter introduced for performance reasons in case when history is not
         * needed.
         *
         * @param integral Vector of symbols with the integral, the first symbol should be
         * `Sym::Integral`
         *
         * @param history An already created `ComputationHistory` object to save the history to.
         * History can be later retrieved by function `get_steps()` or `get_tex_history()`.
         * The dummy object must be provided even if `WITH_HISTORY` is `false`.
         *
         * @return `std::nullopt` if no result has been found, vector of vectors with the solution
         * tree otherwise
         */
        template <bool WITH_HISTORY = true>
        std::optional<std::vector<Symbol>>
        solve_integral_with_history(const std::vector<Symbol>& integral,
                                    ComputationHistory& history) {
            expressions.load_from_vector({single_integral_vacancy()});
            integrals.load_from_vector({first_expression_candidate(integral)});

            for (size_t i = 0;; ++i) {
                simplify_integrals();

                if constexpr (WITH_HISTORY) {
                    history.add_step({expressions.to_vector(), integrals.to_vector(),
                                      ComputationStepType::Simplify});
                }

                check_for_known_integrals();
                apply_known_integrals();

                if constexpr (WITH_HISTORY) {
                    history.add_step({expressions.to_vector(), integrals.to_vector(),
                                      ComputationStepType::ApplySolution});
                }
                if (is_original_expression_solved()) {
                    if constexpr (WITH_HISTORY) {
                        history.complete();
                    }
                    return Collapser::collapse(expressions.to_vector());
                }

                remove_unnecessary_candidates();

                check_heuristics_applicability();
                apply_heuristics();

                if constexpr (WITH_HISTORY) {
                    history.add_step({expressions.to_vector(), integrals.to_vector(),
                                      ComputationStepType::ApplyHeuristic});
                }

                if (has_original_expression_failed()) {
                    return std::nullopt;
                }

                remove_failed_candidates();
            }

            return std::nullopt;
        }

        size_t memory_usage_for_integral(const std::vector<Symbol>& integral,
                                         size_t& total_memory) {
            size_t initial_usage;
            size_t max_usage;
            size_t usage;

            size_t initial_size =
                sizeof(Symbol) *
                    (expressions.symbols_capacity() + integrals.symbols_capacity() +
                     expressions_swap.symbols_capacity() + integrals_swap.symbols_capacity() +
                     help_space.symbols_capacity()) +
                sizeof(size_t) * (scan_array_1.size() + scan_array_2.size()) +
                sizeof(EvaluationStatus) *
                    (evaluation_statuses_1.size() + evaluation_statuses_2.size());
            size_t max_shown_size = initial_size;

            cudaMemGetInfo(&initial_usage, &total_memory);
            max_usage = initial_usage;

            expressions.load_from_vector({single_integral_vacancy()});
            integrals.load_from_vector({first_expression_candidate(integral)});

            max_shown_size = std::max(
                max_shown_size,
                sizeof(Symbol) *
                        (expressions.symbols_capacity() + integrals.symbols_capacity() +
                         expressions_swap.symbols_capacity() + integrals_swap.symbols_capacity() +
                         help_space.symbols_capacity()) +
                    sizeof(size_t) * (scan_array_1.size() + scan_array_2.size()) +
                    sizeof(EvaluationStatus) *
                        (evaluation_statuses_1.size() + evaluation_statuses_2.size()));

            cudaMemGetInfo(&usage, nullptr);
            max_usage = std::min(usage, max_usage);

            for (size_t i = 0;; ++i) {
                simplify_integrals();

                cudaMemGetInfo(&usage, nullptr);
                max_usage = std::min(usage, max_usage);

                max_shown_size = std::max(
                    max_shown_size,
                    sizeof(Symbol) *
                            (expressions.symbols_capacity() + integrals.symbols_capacity() +
                             expressions_swap.symbols_capacity() +
                             integrals_swap.symbols_capacity() + help_space.symbols_capacity()) +
                        sizeof(size_t) * (scan_array_1.size() + scan_array_2.size()) +
                        sizeof(EvaluationStatus) *
                            (evaluation_statuses_1.size() + evaluation_statuses_2.size()));

                check_for_known_integrals();
                apply_known_integrals();

                cudaMemGetInfo(&usage, nullptr);
                max_usage = std::min(usage, max_usage);

                max_shown_size = std::max(
                    max_shown_size,
                    sizeof(Symbol) *
                            (expressions.symbols_capacity() + integrals.symbols_capacity() +
                             expressions_swap.symbols_capacity() +
                             integrals_swap.symbols_capacity() + help_space.symbols_capacity()) +
                        sizeof(size_t) * (scan_array_1.size() + scan_array_2.size()) +
                        sizeof(EvaluationStatus) *
                            (evaluation_statuses_1.size() + evaluation_statuses_2.size()));

                if (is_original_expression_solved()) {
                    break;
                }

                remove_unnecessary_candidates();

                cudaMemGetInfo(&usage, nullptr);
                max_usage = std::min(usage, max_usage);

                max_shown_size = std::max(
                    max_shown_size,
                    sizeof(Symbol) *
                            (expressions.symbols_capacity() + integrals.symbols_capacity() +
                             expressions_swap.symbols_capacity() +
                             integrals_swap.symbols_capacity() + help_space.symbols_capacity()) +
                        sizeof(size_t) * (scan_array_1.size() + scan_array_2.size()) +
                        sizeof(EvaluationStatus) *
                            (evaluation_statuses_1.size() + evaluation_statuses_2.size()));

                check_heuristics_applicability();
                apply_heuristics();

                cudaMemGetInfo(&usage, nullptr);
                max_usage = std::min(usage, max_usage);

                max_shown_size = std::max(
                    max_shown_size,
                    sizeof(Symbol) *
                            (expressions.symbols_capacity() + integrals.symbols_capacity() +
                             expressions_swap.symbols_capacity() +
                             integrals_swap.symbols_capacity() + help_space.symbols_capacity()) +
                        sizeof(size_t) * (scan_array_1.size() + scan_array_2.size()) +
                        sizeof(EvaluationStatus) *
                            (evaluation_statuses_1.size() + evaluation_statuses_2.size()));

                if (has_original_expression_failed()) {
                    break;
                }

                remove_failed_candidates();

                cudaMemGetInfo(&usage, nullptr);
                max_usage = std::min(usage, max_usage);

                max_shown_size = std::max(
                    max_shown_size,
                    sizeof(Symbol) *
                            (expressions.symbols_capacity() + integrals.symbols_capacity() +
                             expressions_swap.symbols_capacity() +
                             integrals_swap.symbols_capacity() + help_space.symbols_capacity()) +
                        sizeof(size_t) * (scan_array_1.size() + scan_array_2.size()) +
                        sizeof(EvaluationStatus) *
                            (evaluation_statuses_1.size() + evaluation_statuses_2.size()));
            }
            printf("Declared bytes:%lu\n", max_shown_size);
            return initial_usage - max_usage;
        }
    };
}

#endif
