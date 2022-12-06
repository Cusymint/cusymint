#ifndef COMPUTATION_STEP_CUH
#define COMPUTATION_STEP_CUH

#include <list>
#include <vector>

#include "Symbol/Symbol.cuh"

namespace Sym {

    using ExprVector = std::vector<std::vector<Sym::Symbol>>;

    enum ComputationStepType { Simplify, Heuristic, Substitution, Solution };

    struct ComputationStep {
        ExprVector expressions;
        ExprVector integrals;
        ComputationStepType step_type;
    };

    using ComputationStepCollection = std::list<ComputationStep>;
    using StepPath = std::list<std::vector<size_t>>;

    class ComputationHistory {
        ComputationStepCollection computation_steps;
        bool completed = false;

        bool is_solution_at_end_of_steps() {
            return !computation_steps.empty() &&
                   computation_steps.back().step_type == ComputationStepType::Solution;
        }

      public:
        void add_step(const ComputationStep& step) { computation_steps.push_back(step); }

        bool is_completed() const { return completed; }

        void complete() {
            if (!is_solution_at_end_of_steps()) {
                Util::crash("ComputationHistory must have solution at the end to complete");
            }

            const auto& last_step = computation_steps.back();
        }
    };
}

#endif