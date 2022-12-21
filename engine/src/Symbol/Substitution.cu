#include "Substitution.cuh"

#include <fmt/core.h>
#include <stdexcept>
#include <vector>

#include "Constants.cuh"
#include "Symbol.cuh"
#include "Symbol/Macros.cuh"

namespace Sym {
    DEFINE_INTO_DESTINATION_OPERATOR(Substitution)
    DEFINE_IDENTICAL_COMPARE_TO(Substitution)
    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(Substitution)
    DEFINE_INVALID_DERIVATIVE(Substitution)

    DEFINE_COMPRESS_REVERSE_TO(Substitution) {
        const size_t new_expression_size = (destination - 1)->size();
        symbol().copy_single_to(*destination);
        destination->as<Substitution>().size = new_expression_size + 1;
    }

    DEFINE_COMPRESSION_SIZE(Substitution) { return 1; }

    DEFINE_ARE_EQUAL(Substitution) {
        return BASE_ARE_EQUAL(Substitution) &&
               symbol->as<Substitution>().substitution_idx == substitution_idx;
    }

    DEFINE_IS_FUNCTION_OF(Substitution) { return results[1]; } // NOLINT
    
    DEFINE_PUSH_CHILDREN_ONTO_STACK(Substitution) {
        if (!is_last_substitution()) {
            stack.push(&next_substitution().symbol());
        }
        stack.push(&expression());
    }

    DEFINE_PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE(Substitution) {
        push_children_onto_stack(stack);
        expression().additional_required_size() += additional_required_size;
    }

    DEFINE_SEAL_WHOLE(Substitution) { size = expression().size() + 1; }

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
        std::vector<Symbol> expr_with_const(expression().size());
        expression().copy_to(*expr_with_const.data());

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
            return fmt::format("{}, {}", next_substitution().to_string(), to_string_this());
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
            return fmt::format("{}, \\quad {}", next_substitution().to_tex(), to_tex_this());
        }

        return to_tex_this();
    }

    __host__ __device__ const Symbol& Substitution::expression() const { return symbol().child(); }

    __host__ __device__ Symbol& Substitution::expression() {
        return const_cast<Symbol&>(const_cast<const Substitution*>(this)->expression());
    }

    __host__ __device__ const Substitution& Substitution::next_substitution() const {
        return (&expression() + expression().size())->as<Substitution>();
    }

    __host__ __device__ Substitution& Substitution::next_substitution() {
        return const_cast<Substitution&>(
            const_cast<const Substitution*>(this)->next_substitution());
    }

    __host__ __device__ bool Substitution::is_last_substitution() const {
        return !(&expression() + expression().size())->is(Type::Substitution);
    }

    __host__ __device__ void Substitution::seal() { size = 1 + expression().size(); }

    std::vector<Symbol> Substitution::substitute(std::vector<Symbol> expr) const {
        if (!is_last_substitution()) {
            expr = next_substitution().substitute(expr);
        }

        std::vector<size_t> variable_indices;
        for (size_t i = 0; i < expr.size(); ++i) {
            if (expr[i].is(Type::Variable)) {
                expr[i].init_from(ExpanderPlaceholder::with_size(expression().size()));
                variable_indices.push_back(i);
            }
        }

        const size_t new_size = expr.size() + variable_indices.size() * (expression().size() - 1);
        // + 1 because compress_to requires that
        std::vector<Symbol> new_expr(new_size + 1);
        auto new_expr_iterator =
            SymbolIterator::from_at(*new_expr.data(), 0, new_expr.size()).good();
        expr.data()->compress_to(new_expr_iterator).unwrap();
        new_expr.resize(new_expr.size() - 1);

        size_t offset = 0;
        for (size_t variable_indice : variable_indices) {
            expression().copy_to(new_expr[variable_indice + offset]);
            offset += expression().size() - 1;
        }

        return new_expr;
    }
}
