#ifndef COMPUTATION_STEP_CUH
#define COMPUTATION_STEP_CUH

#include <list>
#include <string>
#include <tuple>
#include <vector>

#include "Symbol/Symbol.cuh"

namespace Sym {

    using TransformationIdx = size_t;
    using ExprVector = std::vector<std::vector<Symbol>>;
    using OperationList = std::list<std::tuple<TransformationIdx, std::vector<Symbol>, std::vector<Symbol>>>;

    enum ComputationStepType { Simplify, ApplyHeuristic, ApplySolution, SolutionFound };

    const char* get_computation_step_text(ComputationStepType type);

    class ComputationStep {
        ExprVector expression_tree;
        ComputationStepType step_type;

        public:
        ComputationStep(const ExprVector& expressions, const ExprVector& integrals, const ComputationStepType step_type);
        inline ComputationStepType get_step_type() const { return step_type; }
        bool has_solution_path() const;
        ssize_t find_index_in_tree_by_uid(size_t uid);
        void copy_solution_path_from(const ComputationStep& other);
        std::vector<Symbol> get_expression() const;
        OperationList get_operations(const ComputationStep& previous_step);
        void print_step() const;
    };

    using ComputationStepCollection = std::list<ComputationStep>;

    class ComputationHistory {
        ComputationStepCollection computation_steps;
        bool completed = false;

        bool is_solution_at_end_of_steps() {
            return !computation_steps.empty() &&
                   computation_steps.back().get_step_type() == ComputationStepType::ApplySolution;
        }

      public:
        void add_step(const ComputationStep& step) { computation_steps.push_back(step); }

        bool is_completed() const { return completed; }

        void complete();

        std::vector<std::string> get_tex_history() const;

        void print_history() const;
    };
}

#endif