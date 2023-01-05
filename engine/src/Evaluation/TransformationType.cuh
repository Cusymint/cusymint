#ifndef TRANSFORMATION_TYPE_CUH
#define TRANSFORMATION_TYPE_CUH

#include <fmt/core.h>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Symbol/Integral.cuh"
#include "Symbol/Solution.cuh"
#include "Symbol/SubexpressionCandidate.cuh"
#include "Symbol/Substitution.cuh"
#include "Symbol/Symbol.cuh"

namespace Sym {
    class TransformationType {
      public:
        virtual ~TransformationType() = default;
        virtual std::string get_description() const = 0;
        virtual bool equals(const TransformationType& other) const = 0;
    };

    std::optional<std::shared_ptr<TransformationType>>
    get_transformation_type(const SubexpressionCandidate& candidate,
                            const Integral& previous_integral,
                            const std::vector<std::vector<Symbol>> expression_tree);

    class Substitute : public TransformationType {
        std::vector<Symbol> substitution;
        std::vector<Symbol> derivative;
        const std::string variable_name;
        const std::string substitution_name;

      public:
        Substitute(const std::vector<Symbol>& substitution, const std::vector<Symbol> derivative,
                   const size_t& substitution_count) :
            substitution(substitution),
            derivative(derivative),
            variable_name(substitution_count == 1
                              ? "x"
                              : Substitution::nth_substitution_name(substitution_count - 2)),
            substitution_name(Substitution::nth_substitution_name(substitution_count - 1)) {
            if (substitution_count > 1) {
                this->substitution.data()->substitute_variable_with_nth_substitution_name(
                    substitution_count - 2);
                this->derivative.data()->substitute_variable_with_nth_substitution_name(
                    substitution_count - 2);
            }
        }

        std::string get_description() const override {
            return fmt::format(R"(\text{{Substitute}}\: {}={}, \text{{d}} {}={} \text{{d}} {})",
                               substitution_name, substitution.data()->to_tex(), substitution_name,
                               derivative.data()->to_tex(), variable_name);
        }

        bool equals(const TransformationType& other) const override {
            const auto* other_sub = dynamic_cast<const Substitute*>(&other);
            return other_sub != nullptr && substitution == other_sub->substitution &&
                   derivative == other_sub->derivative && variable_name == other_sub->variable_name;
        }
    };

    class SplitSum : public TransformationType {
        std::vector<Symbol> first_term;
        std::vector<Symbol> second_term;

      public:
        SplitSum(const std::vector<Symbol>& first_term, const std::vector<Symbol>& second_term) :
            first_term(first_term), second_term(second_term) {}

        std::string get_description() const override { return "\\text{Split sum}"; }

        bool equals(const TransformationType& other) const override {
            const auto* other_split = dynamic_cast<const SplitSum*>(&other);
            return other_split != nullptr && first_term == other_split->first_term &&
                   second_term == other_split->second_term;
        }
    };

    class IntegrateByParts : public TransformationType {
        std::vector<Symbol> first;
        std::vector<Symbol> second;
        std::vector<Symbol> first_derivative;
        std::vector<Symbol> second_derivative;
        const std::string variable_name;

      public:
        IntegrateByParts(const std::vector<Symbol>& first, const std::vector<Symbol>& second,
                         const std::vector<Symbol>& first_derivative,
                         const std::vector<Symbol>& second_derivative,
                         const size_t& substitution_count) :
            first(first),
            second(second),
            first_derivative(first_derivative),
            second_derivative(second_derivative),
            variable_name(substitution_count > 0
                              ? Substitution::nth_substitution_name(substitution_count - 1)
                              : "x") {
            if (substitution_count > 0) {
                this->first.data()->substitute_variable_with_nth_substitution_name(
                    substitution_count - 1);
                this->second.data()->substitute_variable_with_nth_substitution_name(
                    substitution_count - 1);
                this->first_derivative.data()->substitute_variable_with_nth_substitution_name(
                    substitution_count - 1);
                this->second_derivative.data()->substitute_variable_with_nth_substitution_name(
                    substitution_count - 1);
            }
        }

        std::string get_description() const override {
            return fmt::format(
                R"(\text{{Integrate by parts:}}\: f'({})={},\: g({})={},\: f({})={},\: g'({})={})",
                variable_name, first_derivative.data()->to_tex(), variable_name,
                second.data()->to_tex(), variable_name, first.data()->to_tex(), variable_name,
                second_derivative.data()->to_tex());
        }

        bool equals(const TransformationType& other) const override {
            const auto* other_int = dynamic_cast<const IntegrateByParts*>(&other);
            return other_int != nullptr && first == other_int->first &&
                   second == other_int->second && first_derivative == other_int->first_derivative &&
                   second_derivative == other_int->second_derivative;
        }
    };

    class SolveIntegral : public TransformationType {
        std::vector<Symbol> integral;
        std::vector<Symbol> solution;
        const std::string variable_name;

      public:
        SolveIntegral(const std::vector<Symbol>& integral, const std::vector<Symbol>& solution,
                      const size_t& substitution_count) :
            integral(integral),
            solution(solution),
            variable_name(substitution_count > 0
                              ? Substitution::nth_substitution_name(substitution_count - 1)
                              : "x") {
            if (substitution_count > 0) {
                this->integral.data()->substitute_variable_with_nth_substitution_name(
                    substitution_count - 1);
                this->solution.data()->substitute_variable_with_nth_substitution_name(
                    substitution_count - 1);
            }
        }

        std::string get_description() const override {
            return fmt::format(R"(\text{{Solve integral:}} \int {} \text{{d}} {} = {} + C)",
                               integral.data()->as<Integral>().integrand().to_tex(), variable_name,
                               solution.data()->to_tex());
        }

        bool equals(const TransformationType& other) const override {
            const auto* other_solve = dynamic_cast<const SolveIntegral*>(&other);
            return other_solve != nullptr && integral == other_solve->integral &&
                   solution == other_solve->solution;
        }
    };

    class BringOutConstant : public TransformationType {
        std::vector<Symbol> integral_before;
        std::vector<Symbol> integral_after;

      public:
        std::string get_description() const override { return "\\text{Bring out constant}"; }

        bool equals(const TransformationType& other) const override {
            const auto* other_bring = dynamic_cast<const BringOutConstant*>(&other);
            return other_bring != nullptr && integral_before == other_bring->integral_before &&
                   integral_after == other_bring->integral_after;
        }
    };

    class SimplifyExpression : public TransformationType {
      public:
        std::string get_description() const override { return "\\text{Simplify expression}"; }

        bool equals(const TransformationType& other) const override {
            return dynamic_cast<const SimplifyExpression*>(&other) != nullptr;
        }
    };
}

#endif