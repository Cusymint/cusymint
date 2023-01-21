#include "ComputationHistory.cuh"

#include <cstddef>
#include <exception>
#include <fmt/core.h>
#include <iterator>
#include <memory>
#include <queue>
#include <sys/types.h>
#include <vector>

#include "Evaluation/Collapser.cuh"
#include "Evaluation/IntegratorKernels.cuh"
#include "Symbol/Macros.cuh"
#include "Symbol/MetaOperators.cuh"
#include "Symbol/SubexpressionCandidate.cuh"
#include "Symbol/SubexpressionVacancy.cuh"
#include "Symbol/Symbol.cuh"
#include "Utils/CompileConstants.cuh"
#include "Utils/Cuda.cuh"
#include "Utils/Globalization.cuh"

namespace Sym {
    const char* get_computation_step_text(ComputationStepType type) {
        switch (type) {
        case Simplify:
            return "Simplify expression";
        case ApplyHeuristic:
            return "Apply heuristic";
        case ApplySolution:
            return "Solve integral";
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

            expression_tree.push_back(integrals[i]);
        }
    }

    bool ComputationStep::has_solution_path() const {
        return !expression_tree.empty() &&
               expression_tree[0][0].as<SubexpressionVacancy>().is_solved == 1;
    }

    ssize_t ComputationStep::find_index_in_tree_by_uid(size_t uid) const {
        for (ssize_t i = 0; i < expression_tree.size(); ++i) {
            if (expression_tree[i][0].is(Type::SubexpressionCandidate) &&
                expression_tree[i][0].as<SubexpressionCandidate>().uid == uid) {
                return i;
            }
        }
        return -1;
    }

    const std::vector<Symbol>& ComputationStep::find_by_uid(size_t uid) const {
        const ssize_t idx = find_index_in_tree_by_uid(uid);
        if (idx < 0) {
            Util::crash("No expression of uid=%lu in ComputationStep tree", uid);
        }

        return expression_tree[idx];
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
                this_cand.symbol().at(i)->if_is_do<SubexpressionVacancy>(
                    [&other, &other_cand_symbol = other_cand.symbol(), this, i](auto& vacancy) {
                        const auto& other_vacancy =
                            other_cand_symbol.at(i)->as<SubexpressionVacancy>();
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
    }

    std::vector<Symbol> ComputationStep::get_expression() const {
        if (!has_solution_path()) {
            Util::crash("ComputationStep doesn't have solution path copied from some solution");
        }

        return Collapser::collapse(expression_tree);
    }

    TransformationList ComputationStep::get_operations(const ComputationStep& previous_step) const {
        TransformationList list;

        if (step_type == ComputationStepType::Simplify) {
            list.push_back(std::make_unique<SimplifyExpression>());
            return list;
        }

        size_t max_prev_uid = 0;

        for (const auto& expression : previous_step.expression_tree) {
            if (!expression[0].is(Type::SubexpressionCandidate)) {
                continue;
            }

            const auto& candidate = expression[0].as<SubexpressionCandidate>();

            if (candidate.uid > max_prev_uid) {
                max_prev_uid = candidate.uid;
            }
        }

        std::queue<size_t> index_queue;
        index_queue.push(0);

        while (!index_queue.empty()) {
            const size_t idx = index_queue.front();
            index_queue.pop();
            const auto& expression = expression_tree[idx];
            for (const auto& symbol : expression) {
                if (symbol.is(Type::SubexpressionVacancy)) {
                    if constexpr (Consts::DEBUG) {
                        if (symbol.as<SubexpressionVacancy>().is_solved == 0) {
                            Util::crash(
                                "Unsolved SubexpressionVacancy on a path of a completed step");
                        }
                    }

                    index_queue.push(symbol.as<SubexpressionVacancy>().solver_idx);
                }
            }
            if (!expression[0].is(Type::SubexpressionCandidate)) {
                continue;
            }

            const auto& candidate = expression[0].as<SubexpressionCandidate>();

            if (candidate.uid <= max_prev_uid) {
                continue;
            }

            const auto& creator_integral = previous_step.find_by_uid(candidate.creator_uid)
                                               .data()
                                               ->as<SubexpressionCandidate>()
                                               .arg()
                                               .as<Integral>();

            const auto transformation =
                get_transformation_type(candidate, creator_integral, expression_tree);
            if (transformation.has_value()) {
                list.push_back(transformation.value());
            }
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

    size_t ComputationStep::get_memory_occupance_in_bytes() const {
        size_t size = 0;
        for (const auto& expr : expression_tree) {
            size += expr.size();
        }
        return sizeof(ComputationStep) + size * sizeof(Symbol);
    }

    size_t ComputationHistory::get_memory_occupance_in_bytes() const {
        size_t size = 0;
        for (const auto& step : computation_steps) {
            size += step.get_memory_occupance_in_bytes();
        }
        return sizeof(ComputationHistory) + size;
    }

    void ComputationHistory::complete() {
        if (!is_solution_at_end_of_steps()) {
            Util::crash("ComputationHistory must have solution at the end to complete");
        }

        for (auto it = std::next(computation_steps.rbegin()); it != computation_steps.rend();
             ++it) {
            it->copy_solution_path_from(*std::prev(it));
        }

        completed = true;
    }

    namespace {
        const char* get_ordinal_suffix(size_t trans_idx) {
            trans_idx %= 100;
            if (trans_idx > 10 && trans_idx < 20) {
                return Globalization::TH_SUFFIX;
            }
            switch (trans_idx % 10) {
            case 1:
                return Globalization::ST_SUFFIX;
            case 2:
                return Globalization::ND_SUFFIX;
            case 3:
                return Globalization::RD_SUFFIX;
            default:
                return Globalization::TH_SUFFIX;
            }
        }
    }

    std::vector<std::string> ComputationHistory::get_tex_history() const {
        if (!completed) {
            Util::crash("Trying to get history from uncompleted ComputationHistory");
        }

        const ComputationStep empty_step({}, {}, ComputationStepType::Simplify);

        std::vector<std::string> result;
        std::string prev_str;
        const ComputationStep* prev_step = &empty_step;

        for (const auto& step : computation_steps) {
            const auto expression = step.get_expression();
            const auto expression_str = expression.data()->to_tex();

            if (prev_str != expression_str) {
                const auto& operations = step.get_operations(*prev_step);
                if (operations.size() == 1) {
                    result.push_back(
                        fmt::format(R"(\quad {}:)", operations.front()->get_description()));
                }
                else {
                    size_t trans_idx = 1;
                    for (const auto& trans : step.get_operations(*prev_step)) {
                        result.push_back(fmt::format(R"(\quad \text{{{}{} {}:}}\:{}:)", trans_idx,
                                                     get_ordinal_suffix(trans_idx),
                                                     Globalization::INTEGRAL,
                                                     trans->get_description()));
                        ++trans_idx;
                    }
                }
                result.push_back(fmt::format(R"(=\qquad {})", expression_str));
            }

            prev_str = expression_str;
            prev_step = &step;
        }
        
        if (!result.empty()) {
            result[result.size() - 1] += " + C";    
        }

        if (!result.empty()) {
            result[result.size() - 1] += " + C";
        }

        return result;
    }

    void ComputationHistory::print_history() const {
        for (const auto& step : computation_steps) {
            step.print_step();
        }
    }
}
