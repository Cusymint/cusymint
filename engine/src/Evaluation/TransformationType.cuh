#ifndef TRANSFORMATION_TYPE_CUH
#define TRANSFORMATION_TYPE_CUH

#include <fmt/core.h>
#include <string>
#include <vector>

#include "Symbol/Integral.cuh"
#include "Symbol/Solution.cuh"
#include "Symbol/Substitution.cuh"
#include "Symbol/Symbol.cuh"

namespace Sym {
    class TransformationType {
      public:
        virtual ~TransformationType() = default;
        virtual std::string get_description() const = 0;
    };

    class Substitute : public TransformationType {
        std::vector<Symbol> substitution;
        std::vector<Symbol> derivative;
        const std::string substitution_name;

      public:
        Substitute(const std::vector<Symbol>& substitution, const std::vector<Symbol> derivative,
                   const size_t& substitution_count) :
            substitution(substitution),
            derivative(derivative),
            substitution_name(Substitution::nth_substitution_name(substitution_count - 1)) {
            this->substitution.data()->substitute_variable_with_nth_substitution_name(
                substitution_count - 1);
            this->derivative.data()->substitute_variable_with_nth_substitution_name(
                substitution_count - 1);
        }

        std::string get_description() const override {
            return fmt::format("\\text{{Substitute}}\\: {}={}, \\dd {}={}", substitution_name,
                               substitution.data()->to_tex(), substitution_name,
                               derivative.data()->to_tex());
        }
    };

    class SplitSum : public TransformationType {
        std::vector<Symbol> first_term;
        std::vector<Symbol> second_term;

      public:
        SplitSum(const std::vector<Symbol>& first_term, const std::vector<Symbol>& second_term) :
            first_term(first_term), second_term(second_term) {}

        std::string get_description() const override { return "\\text{Split sum}"; }
    };

    class IntegrateByParts : public TransformationType {
        std::vector<Symbol> first_factor;

      public:
        std::string get_description() const override { return "\\text{Integrate by parts}"; }
    };

    class SolveIntegral : public TransformationType {
        std::vector<Symbol> integral;
        std::vector<Symbol> solution;

      public:
        SolveIntegral(const std::vector<Symbol>& integral, const std::vector<Symbol>& solution,
                      const size_t& substitution_count) :
            integral(integral), solution(solution) {
            if (substitution_count > 0) {
                this->integral.data()->substitute_variable_with_nth_substitution_name(
                    substitution_count - 1);
                this->solution.data()->substitute_variable_with_nth_substitution_name(
                    substitution_count - 1);
            }
        }

        std::string get_description() const override {
            return fmt::format("\\text{{Solve integral:}} {} = {}", integral.data()->to_tex(),
                               solution.data()->to_tex());
        }
    };

    class BringOutConstant : public TransformationType {
        std::vector<Symbol> integral_before;
        std::vector<Symbol> integral_after;

      public:
        std::string get_description() const override { return "\\text{Bring out constant}"; }
    };

    class SimplifyExpression : public TransformationType {
      public:
        std::string get_description() const override { return "\\text{Simplify expression}"; }
    };
}

#endif