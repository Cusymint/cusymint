#include "ComputationHistory.cuh"
#include "Evaluation/Collapser.cuh"
#include "Symbol/SubexpressionCandidate.cuh"
#include "Symbol/SubexpressionVacancy.cuh"
#include "Symbol/Symbol.cuh"
#include "Utils/Cuda.cuh"
#include <fmt/core.h>
#include <iterator>
#include <vector>

namespace Sym {
    const char* get_computation_step_text(ComputationStepType type) {
        switch (type) {
        case Simplify:
            return "Simplify expression:";
        case ApplyHeuristic:
            return "Apply heuristic:";
        case ApplySolution:
            return "Solve integral:";
        case SolutionFound:
            return "Solution:";
        default:
            return "(invalid operation)";
        }
    }

    ComputationStep::ComputationStep(const ExprVector& expressions, const ExprVector& integrals,
                                     const ComputationStepType step_type) :
        expression_tree(expressions), step_type(step_type) {
        for (size_t i = 0; i < integrals.size(); ++i) {
            const auto& candidate = integrals[i][0].as<SubexpressionCandidate>();

            auto& vacancy = expression_tree[candidate.vacancy_expression_idx][candidate.vacancy_idx]
                                .as<SubexpressionVacancy>();

            if (vacancy.is_solved != 1) {
                vacancy.is_solved = 1;
                vacancy.solver_idx = expressions.size() + i;
            }

            expression_tree.push_back(integrals[i]);
        }
    }

    bool ComputationStep::has_solution_path() const {
        return !expression_tree.empty() &&
               expression_tree[0][0].as<SubexpressionVacancy>().is_solved == 1;
    }

    void ComputationStep::copy_solution_path_from(const ComputationStep& other) {
        if (other.step_type != ComputationStepType::ApplySolution) {
            Util::crash("Trying to copy path from ComputationStep which is not Solution");
        }

        for (size_t i = 0; i < expression_tree.size(); ++i) {
            for (size_t j = 0; j < expression_tree[i].size(); ++j) {
                expression_tree[i][j].if_is_do<SubexpressionVacancy>(
                    [&other_vacancy = other.expression_tree[i][j]](SubexpressionVacancy& vacancy) {
                        if (other_vacancy.as<SubexpressionVacancy>().is_solved == 1 &&
                            vacancy.is_solved != 1) {
                            vacancy.is_solved = 1;
                            vacancy.solver_idx =
                                other_vacancy.as<SubexpressionVacancy>().solver_idx;
                        }
                    });
            }
        }
    }

    std::vector<Symbol> ComputationStep::get_expression() const {
        if (!has_solution_path()) {
            Util::crash("ComputationStep doesn't have solution path copied from some solution");
        }

        return Collapser::collapse(expression_tree);
    }

    void ComputationHistory::complete() {
        if (!is_solution_at_end_of_steps()) {
            Util::crash("ComputationHistory must have solution at the end to complete");
        }

        const auto last_step_it = std::prev(computation_steps.end());
        for (auto it = computation_steps.begin(); it != last_step_it; ++it) {
            it->copy_solution_path_from(*last_step_it);
        }

        completed = true;
    }

    std::vector<std::string> ComputationHistory::get_tex_history() const {
        if (!completed) {
            Util::crash("Trying to get history from uncompleted ComputationHistory");
        }

        std::vector<std::string> result;
        std::string prev_str;

        for (const auto& step : computation_steps) {
            const auto expression = step.get_expression();
            const auto expression_str = expression.data()->to_tex();

            if (prev_str != expression_str) {
                result.push_back(fmt::format(R"(\text{{ {} }} \quad {})",
                                             get_computation_step_text(step.get_step_type()),
                                             expression.data()->to_tex()));
            }

            prev_str = expression_str;
        }

        return result;
    }
}