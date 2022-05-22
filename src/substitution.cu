#include "substitution.cuh"

#include <stdexcept>
#include <vector>

#include "constants.cuh"
#include "symbol.cuh"

namespace Sym {
    DEFINE_UNSUPPORTED_COMPRESS_REVERSE_TO(Substitution)
    DEFINE_INTO_DESTINATION_OPERATOR(Substitution)

    DEFINE_COMPARE(Substitution) {
        return BASE_COMPARE(Substitution) &&
               symbol->substitution.substitution_idx == substitution_idx;
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
        return nth_substitution_name(substitution_idx) + " = " +
               expr_with_const.data()->to_string();
    }

    std::string Substitution::to_string() const {
        std::string sub_substitutions;
        if (!is_last_substitution()) {
            sub_substitutions = next_substitution()->to_string() + ", ";
        }

        return sub_substitutions + to_string_this();
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
        std::copy(integral.begin(), integral.begin() + integrand_offset, new_integral.begin());

        Substitution::create(substitution_expr.data(), new_integral.data() + integrand_offset,
                             integral.data()->integral.substitution_count);

        new_integral.data()->integral.substitution_count += 1;
        new_integral.data()->integral.integrand_offset += 1 + substitution_expr.size();
        new_integral.data()->integral.size += 1 + substitution_expr.size();

        integral.data()->integrand()->copy_to(new_integral.data()->integrand());

        return new_integral;
    }
}
