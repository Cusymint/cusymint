#ifndef COMPUTATION_STEP_CUH
#define COMPUTATION_STEP_CUH

#include <list>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "Symbol/Symbol.cuh"
#include "TransformationType.cuh"

namespace Sym {

    using ExprVector = std::vector<std::vector<Symbol>>;
    using TransformationList = std::list<std::shared_ptr<TransformationType>>;

    enum ComputationStepType { Simplify, ApplyHeuristic, ApplySolution };

    const char* get_computation_step_text(ComputationStepType type);

    class ComputationStep {
        ExprVector expression_tree;
        ComputationStepType step_type;

        public:
        ComputationStep(const ExprVector& expressions, const ExprVector& integrals, const ComputationStepType step_type);
        inline ComputationStepType get_step_type() const { return step_type; }
        bool has_solution_path() const;
        ssize_t find_index_in_tree_by_uid(size_t uid) const;
        const std::vector<Symbol>& find_by_uid(size_t uid) const;
        void copy_solution_path_from(const ComputationStep& other);
        std::vector<Symbol> get_expression() const;
        TransformationList get_operations(const ComputationStep& previous_step) const;
        void print_step() const;
        size_t get_memory_occupance_in_bytes() const;
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

        const ComputationStepCollection& get_steps() const { return computation_steps; } 

        size_t get_memory_occupance_in_bytes() const;
    };
}

#endif