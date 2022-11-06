#include "Substitution.cuh"

#include <fmt/core.h>
#include <stdexcept>
#include <vector>

#include "Constants.cuh"
#include "Symbol.cuh"
#include "Symbol/Macros.cuh"

namespace Sym {
    DEFINE_INTO_DESTINATION_OPERATOR(Substitution)

    DEFINE_COMPRESS_REVERSE_TO(Substitution) {
        const size_t new_expression_size = (destination - 1)->size();
        symbol()->copy_single_to(destination);
        destination->substitution.size = new_expression_size + 1;
        return 1;
    }

    __host__ __device__ size_t
    Substitution::compress_reverse_substitutions_to(Symbol* const destination) {
        size_t offset = 0;
        if (!is_last_substitution()) {
            offset = next_substitution()->compress_reverse_substitutions_to(destination);
        }

        const size_t new_substitution_size = symbol()->compress_reverse_to(destination + offset);
        return new_substitution_size + offset;
    }

    DEFINE_COMPARE(Substitution) {
        return BASE_COMPARE(Substitution) &&
               symbol->substitution.substitution_idx == substitution_idx;
    }

    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(Substitution)

    DEFINE_IS_FUNCTION_OF(Substitution) {
        return expression()->is_function_of(expressions, expression_count);
    } // NOLINT

    DEFINE_PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE(Substitution) {
        stack.push(expression());
        expression()->additional_required_size() += additional_required_size;
    }

    const char* const Substitution::SUBSTITUTION_NAMES[] = {"u", "v", "w", "t"};
    const size_t Substitution::SUBSTITUTION_NAME_COUNT =
        sizeof(Substitution::SUBSTITUTION_NAMES) / sizeof(Substitution::SUBSTITUTION_NAMES[0]);

    std::string Substitution::nth_substitution_name(const size_t n) {
        if (n < SUBSTITUTION_NAME_COUNT) {
            return SUBSTITUTION_NAMES[n];
        }

        std::string name = SUBSTITUTION_NAMES[SUBSTITUTION_NAME_COUNT - 1] + std::string("_") +
                           std::to_string(n - SUBSTITUTION_NAME_COUNT);

        if (name.size() + 1 > UnknownConstant::NAME_LEN) {
            name = "u_?";
        }

        return name;
    }

    __host__ __device__ void Substitution::create(const Symbol* const expression,
                                                  Symbol* const destination,
                                                  const size_t substitution_idx) {
        Substitution* const substitution = destination << Substitution::builder();
        expression->copy_to(substitution->expression());
        substitution->substitution_idx = substitution_idx;
        substitution->seal();
    }

    std::vector<Symbol> Substitution::expression_with_constant() const {
        std::vector<Symbol> expr_with_const(expression()->size());
        expression()->copy_to(expr_with_const.data());

        if (substitution_idx > 0) {
            expr_with_const.data()->substitute_variable_with_nth_substitution_name(
                substitution_idx - 1);
        }

        return expr_with_const;
    }

    std::string Substitution::to_string_this() const {
        std::vector<Symbol> expr_with_const = expression_with_constant();
        return fmt::format("{} = {}", nth_substitution_name(substitution_idx),
                           expr_with_const.data()->to_string());
    }

    std::string Substitution::to_string() const {
        if (!is_last_substitution()) {
            return fmt::format("{}, {}", next_substitution()->to_string(), to_string_this());
        }

        return to_string_this();
    }

    std::string Substitution::to_tex_this() const {
        std::vector<Symbol> expr_with_const = expression_with_constant();
        return fmt::format("{} = {}", nth_substitution_name(substitution_idx),
                           expr_with_const.data()->to_tex());
    }

    std::string Substitution::to_tex() const {
        if (!is_last_substitution()) {
            return fmt::format("{}, \\quad {}", next_substitution()->to_tex(), to_tex_this());
        }

        return to_tex_this();
    }

    __host__ __device__ Symbol* Substitution::expression() { return Symbol::from(this)->child(); }

    __host__ __device__ const Symbol* Substitution::expression() const {
        return Symbol::from(this)->child();
    }

    __host__ __device__ Substitution* Substitution::next_substitution() {
        return &(expression() + expression()->size())->substitution;
    }

    __host__ __device__ const Substitution* Substitution::next_substitution() const {
        return &(expression() + expression()->size())->substitution;
    }

    __host__ __device__ bool Substitution::is_last_substitution() const {
        return !(expression() + expression()->size())->is(Type::Substitution);
    }

    __host__ __device__ void Substitution::seal() { size = 1 + expression()->size(); }

    std::vector<Symbol> substitute(const std::vector<Symbol>& integral,
                                   const std::vector<Symbol>& substitution_expr) {
        if (!integral.data()->is(Type::Integral)) {
            throw std::invalid_argument("Explicit substitutions are only allowed in integrals");
        }

        std::vector<Symbol> new_integral(integral.size() + substitution_expr.size() + 1);

        const size_t integrand_offset = integral.data()->integral.integrand_offset;
        std::copy(integral.data(), integral.data() + integrand_offset, new_integral.begin());

        Substitution::create(substitution_expr.data(), new_integral.data() + integrand_offset,
                             integral.data()->integral.substitution_count);

        new_integral.data()->integral.substitution_count += 1;
        new_integral.data()->integral.integrand_offset += 1 + substitution_expr.size();
        new_integral.data()->integral.size += 1 + substitution_expr.size();

        integral.data()->integrand()->copy_to(new_integral.data()->integrand());

        return new_integral;
    }
}
