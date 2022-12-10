#include "ComputationHistory.cuh"
#include "Evaluation/Collapser.cuh"
#include "Symbol/SubexpressionCandidate.cuh"
#include "Symbol/SubexpressionVacancy.cuh"
#include "Symbol/Symbol.cuh"
#include "Utils/Cuda.cuh"
#include <exception>
#include <fmt/core.h>
#include <iterator>
#include <sys/types.h>
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

            // if (vacancy.is_solved != 1) {
            //     vacancy.is_solved = 1;
            //     vacancy.solver_idx = expressions.size() + i;
            // }

            expression_tree.push_back(integrals[i]);
        }
    }

    bool ComputationStep::has_solution_path() const {
        return !expression_tree.empty() &&
               expression_tree[0][0].as<SubexpressionVacancy>().is_solved == 1;
    }

    ssize_t ComputationStep::find_index_in_tree_by_uid(size_t uid) {
        for (ssize_t i = 0; i < expression_tree.size(); ++i) {
            if (expression_tree[i][0].is(Type::SubexpressionCandidate) &&
                expression_tree[i][0].as<SubexpressionCandidate>().uid == uid) {
                return i;
            }
        }
        return -1;
    }

    void ComputationStep::copy_solution_path_from(const ComputationStep& other) {
        auto this_it = expression_tree.begin();
        auto other_it = other.expression_tree.cbegin();

        auto& initial_vacancy = this_it->data()->as<SubexpressionVacancy>();
        const auto& other_initial_vacancy = other_it->data()->as<SubexpressionVacancy>();

        if (other_initial_vacancy.is_solved != 1) {
            Util::crash("Trying to copy path from ComputationStep which is not solved nor has "
                        "copied solution path");
        }

        const auto& candidate =
            other.expression_tree[other_initial_vacancy.solver_idx][0].as<SubexpressionCandidate>();
        ssize_t index = find_index_in_tree_by_uid(candidate.uid);
        if (index < 0) {
            index = find_index_in_tree_by_uid(candidate.creator_uid);
        }
        if (index < 0) {
            Util::crash("No expression of uid=%lu nor creator_uid=%lu in "
                        "ComputationStep tree",
                        candidate.uid, candidate.creator_uid);
        }
        initial_vacancy.is_solved = 1;
        initial_vacancy.solver_idx = index;

        ++this_it;
        ++other_it;

        while (this_it != expression_tree.end() && other_it != other.expression_tree.cend()) {
            if (!this_it->data()->is(Type::SubexpressionCandidate)) {
                ++this_it;
                continue;
            }
            if (!other_it->data()->is(Type::SubexpressionCandidate)) {
                ++other_it;
                continue;
            }

            auto& this_cand = this_it->data()->as<SubexpressionCandidate>();
            const auto& other_cand = other_it->data()->as<SubexpressionCandidate>();

            if (this_cand.uid < other_cand.uid) {
                ++this_it;
                continue;
            }
            if (this_cand.uid > other_cand.uid) {
                ++other_it;
                continue;
            }

            for (int i = 0; i < this_cand.size; ++i) {
                this_cand.symbol()->at(i)->if_is_do<SubexpressionVacancy>(
                    [&other, other_cand_symbol = other_cand.symbol(), this, i](auto& vacancy) {
                        const auto& other_vacancy = other_cand_symbol->at(i)->as<SubexpressionVacancy>();
                        if (other_vacancy.is_solved == 1 && vacancy.is_solved != 1) {
                            const auto& candidate =
                                other.expression_tree[other_vacancy.solver_idx][0]
                                    .as<SubexpressionCandidate>();
                            ssize_t index = find_index_in_tree_by_uid(candidate.uid);
                            if (index < 0) {
                                index = find_index_in_tree_by_uid(candidate.creator_uid);
                            }
                            if (index < 0) {
                                Util::crash("No expression of uid=%lu nor creator_uid=%lu in "
                                            "ComputationStep tree",
                                            candidate.uid, candidate.creator_uid);
                            }
                            vacancy.is_solved = 1;
                            vacancy.solver_idx = index;
                        }
                    });
            }
            ++this_it;
            ++other_it;
        }
        print_step();
        // for (size_t i = 0; i < expression_tree.size(); ++i) {
        //     for (size_t j = 0; j < expression_tree[i].size(); ++j) {
        //         expression_tree[i][j].if_is_do<SubexpressionVacancy>(
        //             [&other_vacancy = other.expression_tree[i][j]](SubexpressionVacancy& vacancy)
        //             {
        //                 if (other_vacancy.as<SubexpressionVacancy>().is_solved == 1 &&
        //                     vacancy.is_solved != 1) {
        //                     vacancy.is_solved = 1;
        //                     vacancy.solver_idx =
        //                         other_vacancy.as<SubexpressionVacancy>().solver_idx;
        //                 }
        //             });
        //     }
        // }
    }

    std::vector<Symbol> ComputationStep::get_expression() const {
        if (!has_solution_path()) {
            Util::crash("ComputationStep doesn't have solution path copied from some solution");
        }

        return Collapser::collapse(expression_tree);
    }

    OperationList ComputationStep::get_operations(const ComputationStep& previous_step) {
        OperationList list;

        auto this_it = expression_tree.cbegin();
        auto prev_step_it = previous_step.expression_tree.cbegin();

        while (prev_step_it != previous_step.expression_tree.cend()) {
            if (Symbol::are_expressions_equal(*this_it->data(), *prev_step_it->data())) {
                ++this_it;
            }
            ++prev_step_it;
        }

        while (this_it != expression_tree.cend()) {
            const auto& candidate = this_it->data()->as<SubexpressionCandidate>();
        }

        return list;
    }

    void ComputationStep::print_step() const {
        fmt::print("{{\n  Step type: {}\n  Expression tree:\n",
                   get_computation_step_text(step_type));
        for (const auto& expr : expression_tree) {
            fmt::print("    {}\n", expr.data()->to_string());
        }
        fmt::print("}}\n");
    }

    void ComputationHistory::complete() {
        if (!is_solution_at_end_of_steps()) {
            Util::crash("ComputationHistory must have solution at the end to complete");
        }

        // const auto last_step_it = std::prev(computation_steps.end());
        // for (auto it = computation_steps.begin(); it != last_step_it; ++it) {
        //     it->copy_solution_path_from(*last_step_it);
        // }
        for (auto it = std::next(computation_steps.rbegin()); it != computation_steps.rend();
             ++it) {
            it->copy_solution_path_from(*std::prev(it));
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

    void ComputationHistory::print_history() const {
        for (const auto& step : computation_steps) {
            step.print_step();
        }
    }
}