#ifndef TRANSFORMATION_TYPE_CUH
#define TRANSFORMATION_TYPE_CUH

#include <fmt/core.h>
#include <string>
#include <vector>

#include "Symbol/Integral.cuh"
#include "Symbol/Solution.cuh"
#include "Symbol/Symbol.cuh"

namespace Sym {
    class TransformationType {
      public:
        virtual ~TransformationType() = default;
        virtual std::string get_description() const = 0;
    };

    class Substitute : public TransformationType {
        const std::vector<Symbol> substitution;
        const std::vector<Symbol> derivative;

      public:
        Substitute(const std::vector<Symbol>& substitution, const std::vector<Symbol> derivative) :
            substitution(substitution), derivative(derivative) {}

        std::string get_description() const override {
            return fmt::format("Substitute {}, {}", substitution.data()->to_tex(),
                               derivative.data()->to_tex());
        }
    };

    class SplitSum : public TransformationType {
        std::vector<Symbol> first_term;
        std::vector<Symbol> second_term;

      public:
        SplitSum(const std::vector<Symbol>& first_term, const std::vector<Symbol>& second_term) :
            first_term(first_term), second_term(second_term) {}

        std::string get_description() const override { return "Split sum"; }
    };

    class IntegrateByParts : public TransformationType {
        std::vector<Symbol> first_factor;

      public:
        std::string get_description() const override { return "Integrate by parts"; }
    };

    class SolveIntegral : public TransformationType {
        std::vector<Symbol> integral;
        std::vector<Symbol> solution;

      public:
        SolveIntegral(const std::vector<Symbol> integral, const std::vector<Symbol> solution) :
            integral(integral), solution(solution) {}

        std::string get_description() const override {
            return fmt::format("Solve integral: {} = {}", integral.data()->to_tex(),
                               solution.data()->to_tex());
        }
    };

    class BringOutConstant : public TransformationType {
        std::vector<Symbol> integral_before;
        std::vector<Symbol> integral_after;

      public:
        std::string get_description() const override { return "Bring out constant"; }
    };

    class SimplifyExpression : public TransformationType {
        public:
        std::string get_description() const override { return "Simplify expression"; }
    }
}

#endif