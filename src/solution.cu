#include "solution.cuh"

#include <stdexcept>

#include "symbol.cuh"

namespace Sym {
    DEFINE_UNSUPPORTED_COMPRESS_REVERSE_TO(Solution)
    DEFINE_INTO_DESTINATION_OPERATOR(Solution)

    DEFINE_COMPARE(Solution) {
        return BASE_COMPARE(Solution) &&
               symbol->solution.substitution_count == substitution_count &&
               symbol->solution.expression_offset == expression_offset;
    }

    __host__ __device__ void Solution::seal_no_substitutions() { seal_substitutions(0, 0); }

    __host__ __device__ void Solution::seal_single_substitution() {
        seal_substitutions(1, (Symbol::from(this) + expression_offset)->size());
    }

    __host__ __device__ void Solution::seal_substitutions(const size_t count, const size_t size) {
        expression_offset = 1 + size;
        substitution_count = count;
    }

    __host__ __device__ void Solution::seal() {
        size = expression_offset + expression()->size();
    }

    __host__ __device__ Symbol* Solution::expression() {
        return Symbol::from(this) + expression_offset;
    }

    __host__ __device__ const Symbol* Solution::expression() const {
        return Symbol::from(this) + expression_offset;
    }

    __host__ __device__ const Substitution* Solution::first_substitution() const {
        return &Symbol::from(this)->child()->substitution;
    }

    __host__ __device__ Substitution* Solution::first_substitution() {
        return &Symbol::from(this)->child()->substitution;
    }

    __host__ __device__ size_t Solution::substitutions_size() const {
        return size - 1 - expression()->size();
    };

    std::string Solution::to_string() const {
        std::vector<Symbol> expression_copy(expression()->size());
        expression()->copy_to(expression_copy.data());

        std::string substitutions_str;
        if (substitution_count != 0) {
            substitutions_str = ", " + first_substitution()->to_string();
            expression_copy.data()->substitute_variable_with_nth_substitution_name(
                substitution_count - 1);
        }

        return "Solution: " + expression_copy.data()->to_string() + substitutions_str;
    }

    std::vector<Symbol> solution(const std::vector<Symbol>& arg) {
        std::vector<Symbol> res(arg.size() + 1);

        Solution* const solution = res.data() << Solution::builder();
        solution->seal_no_substitutions();
        arg.data()->copy_to(solution->expression());
        solution->seal();

        return res;
    }
}
