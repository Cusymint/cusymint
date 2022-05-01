#include "substitution.cuh"

#include <stdexcept>
#include <vector>

#include "constants.cuh"
#include "symbol.cuh"

namespace Sym {
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
            throw std::length_error("Substitution variable name is too long");
        }

        return name;
    }

    std::string Substitution::to_string() {
        Symbol* symbols = Symbol::from(this) + 1;
        std::vector<Symbol> subst_symbols(symbols->unknown.total_size);
        std::copy(symbols, symbols + symbols->unknown.total_size, subst_symbols.data());

        if (substitution_idx > 0) {
            Symbol substitute;
            substitute.unknown_constant =
                UnknownConstant::create(nth_substitution_name(substitution_idx - 1).c_str());

            subst_symbols.data()->substitute_variable_with(substitute);
        }

        std::string sub_substitutions;

        if (sub_substitution_count != 0) {
            sub_substitutions = next_substitution()->to_string() + ", ";
        }

        return sub_substitutions + nth_substitution_name(substitution_idx) + " = " +
               subst_symbols.data()->to_string();
    }

    __host__ __device__ Symbol* Substitution::next_substitution() {
        Symbol* this_symbol = Symbol::from(this);
        return this_symbol + 1 + this_symbol[1].unknown.total_size;
    }

    std::vector<Symbol> substitute(const std::vector<Symbol>& integral,
                                   const std::vector<Symbol>& substitution_expr) {
        if (!integral.data()->is(Type::Integral)) {
            throw std::invalid_argument("Explicit substitutions are only allowed in integrals");
        }

        std::vector<Symbol> subst_integral(integral.size() + substitution_expr.size() + 1);

        size_t integrand_offset = integral[0].integral.integrand_offset;
        std::copy(integral.begin(), integral.begin() + integrand_offset, subst_integral.begin());

        subst_integral[0].integral.substitution_count += 1;
        subst_integral[0].integral.integrand_offset += 1 + substitution_expr.size();
        subst_integral[0].integral.total_size += 1 + substitution_expr.size();

        Symbol* current_substitution = subst_integral.data() + 1;
        for (size_t i = 0; i < integral[0].integral.substitution_count; ++i) {
            current_substitution->substitution.sub_substitution_count += 1;
            current_substitution->substitution.total_size += 1 + substitution_expr.size();
            current_substitution = current_substitution->substitution.next_substitution();
        }

        current_substitution->substitution = Substitution::create();
        current_substitution->substitution.total_size =
            1 + substitution_expr.size() + integral[0].unknown.total_size;
        current_substitution->substitution.substitution_idx =
            integral[0].integral.substitution_count;
        current_substitution->substitution.sub_substitution_count = 0;

        std::copy(substitution_expr.begin(), substitution_expr.end(), current_substitution + 1);

        std::copy(integral.begin() + integral[0].integral.integrand_offset,
                  integral.begin() + integral[0].integral.total_size,
                  subst_integral.begin() + subst_integral[0].integral.integrand_offset);

        return subst_integral;
    }
}
