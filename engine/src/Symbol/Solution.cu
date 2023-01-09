#include "Solution.cuh"

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "Symbol.cuh"
#include "Symbol/Macros.cuh"

namespace Sym {
    DEFINE_INTO_DESTINATION_OPERATOR(Solution)
    DEFINE_IDENTICAL_COMPARE_TO(Solution)
    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(Solution)
    DEFINE_INVALID_DERIVATIVE(Solution)
    DEFINE_INVALID_SEAL_WHOLE(Solution)

    DEFINE_COMPRESS_REVERSE_TO(Solution) {
        size_t new_substitutions_size = 0;
        Symbol* substitution = destination - 1;
        // we assume that substitutions do not need additional size (this is naive imo)
        for (size_t index = substitution_count; index > 0; --index) {
            new_substitutions_size += substitution->size();
            substitution -= substitution->size();
        }

        substitution->size() += substitution->additional_required_size();
        substitution->additional_required_size() = 0;

        const size_t new_expression_size = substitution->size();
        symbol().copy_single_to(*destination);
        destination->as<Solution>().size = new_expression_size + new_substitutions_size + 1;
        destination->as<Solution>().expression_offset = new_substitutions_size + 1;
    }

    DEFINE_COMPRESSION_SIZE(Solution) { return 1; }

    DEFINE_ARE_EQUAL(Solution) {
        return BASE_ARE_EQUAL(Solution) &&
               symbol->as<Solution>().substitution_count == substitution_count &&
               symbol->as<Solution>().expression_offset == expression_offset &&
               symbol->as<Solution>().solved_by_known_integral == solved_by_known_integral;
    }

    DEFINE_IS_FUNCTION_OF(Solution) { return results[expression_offset]; } // NOLINT

    DEFINE_PUSH_CHILDREN_ONTO_STACK(Solution) {
        if (substitution_count > 0) {
            stack.push(&first_substitution().symbol());
        }

        stack.push(&expression());
    }

    DEFINE_PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE(Solution) {
        push_children_onto_stack(stack);
        expression().additional_required_size() += additional_required_size;
    }

    __host__ __device__ void Solution::seal_no_substitutions() { seal_substitutions(0, 0); }

    __host__ __device__ void Solution::seal_single_substitution() {
        seal_substitutions(1, (&symbol() + expression_offset)->size());
    }

    __host__ __device__ void Solution::seal_substitutions(const size_t count, const size_t size) {
        expression_offset = 1 + size;
        substitution_count = count;
    }

    __host__ __device__ void Solution::seal() { size = expression_offset + expression().size(); }

    __host__ __device__ const Symbol& Solution::expression() const {
        return symbol()[expression_offset];
    }

    __host__ __device__ Symbol& Solution::expression() {
        return const_cast<Symbol&>(const_cast<const Solution*>(this)->expression());
    }

    __host__ __device__ const Substitution& Solution::first_substitution() const {
        return symbol().child().as<Substitution>();
    }

    __host__ __device__ Substitution& Solution::first_substitution() {
        return const_cast<Substitution&>(const_cast<const Solution*>(this)->first_substitution());
    }

    __host__ __device__ size_t Solution::substitutions_size() const {
        return size - 1 - expression().size();
    };

    std::string Solution::to_string() const {
        std::vector<Symbol> expression_copy(expression().size());
        expression().copy_to(*expression_copy.data());

        if (substitution_count != 0) {
            expression_copy.data()->substitute_variable_with_nth_substitution_name(
                substitution_count - 1);
            return fmt::format("Solution: {}, {}", expression_copy.data()->to_string(),
                               first_substitution().to_string());
        }

        return fmt::format("Solution: {}", expression_copy.data()->to_string());
    }

    std::string Solution::to_tex() const {
        std::vector<Symbol> expression_copy(expression().size());
        expression().copy_to(*expression_copy.data());

        if (substitution_count != 0) {
            expression_copy.data()->substitute_variable_with_nth_substitution_name(
                substitution_count - 1);
            return fmt::format(R"(\text{{Solution: }} {}, \quad {})",
                               expression_copy.data()->to_tex(), first_substitution().to_tex());
        }

        return fmt::format(R"(\text{{Solution: }} {})", expression_copy.data()->to_tex());
    }

    std::vector<Symbol> Solution::substitute_substitutions() const {
        std::vector<Symbol> this_expression(expression().size());
        std::copy(&expression(), &expression() + expression().size(), this_expression.begin());

        if (substitution_count == 0) {
            return this_expression;
        }

        return first_substitution().substitute(this_expression);
    }

    std::vector<Symbol> solution(const std::vector<Symbol>& arg, bool solved_by_known_integral) {
        std::vector<Symbol> res(arg.size() + 1);

        Solution* const solution = res.data() << Solution::builder();
        solution->seal_no_substitutions();
        arg.data()->copy_to(solution->expression());
        solution->solved_by_known_integral = solved_by_known_integral;
        solution->seal();

        return res;
    }

    std::vector<Symbol> solution(const std::vector<Symbol>& arg,
                                 const std::vector<std::vector<Symbol>>& substitutions, bool solved_by_known_integral) {
        size_t res_size = arg.size() + 1;
        for (const auto& sub : substitutions) {
            res_size += sub.size() + 1;
        }
        std::vector<Symbol> res(res_size);
        Solution* const solution = res.data() << Solution::builder();
        Symbol* current_dst = &res.data()->child();
        for (size_t i = 0; i < substitutions.size(); ++i) {
            Substitution::create(substitutions[i].data(), current_dst, i);
            current_dst += current_dst->size();
        }
        solution->seal_substitutions(substitutions.size(), current_dst - &res.data()->child());
        arg.data()->copy_to(solution->expression());
        solution->solved_by_known_integral = solved_by_known_integral;
        solution->seal();

        return res;
    }
}
