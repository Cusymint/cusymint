#include "integral.cuh"

#include "substitution.cuh"
#include "symbol.cuh"

namespace Sym {
    DEFINE_UNSUPPORTED_COMPRESS_REVERSE_TO(Integral)

    DEFINE_COMPARE(Integral) {
        return BASE_COMPARE(Integral) &&
               symbol->integral.substitution_count == substitution_count &&
               symbol->integral.integrand_offset == integrand_offset;
    }

    std::string Integral::to_string() const {
        std::string last_substitution_name;
        std::string substitutions_str;

        const Symbol* const integrand = Integral::integrand();
        std::vector<Symbol> subst_symbols(integrand->total_size());
        std::copy(integrand, integrand + integrand->total_size(), subst_symbols.data());

        if (substitution_count == 0) {
            last_substitution_name = "x";
        }
        else {
            last_substitution_name = Substitution::nth_substitution_name(substitution_count - 1);
            const Symbol* const first_substitution = Symbol::from(this) + 1;
            substitutions_str = ", " + first_substitution->to_string();

            Symbol substitute;
            substitute.unknown_constant = UnknownConstant::create(last_substitution_name.c_str());
            subst_symbols[0].substitute_variable_with(substitute);
        }

        return "∫" + subst_symbols[0].to_string() + "d" + last_substitution_name +
               substitutions_str;
    }

    __host__ __device__ Symbol* Integral::integrand() {
        return Symbol::from(this) + integrand_offset;
    }

    __host__ __device__ const Symbol* Integral::integrand() const {
        return Symbol::from(this) + integrand_offset;
    }

    __host__ __device__ void
    Integral::copy_substitutions_with_an_additional_one(const Symbol* const substitution,
                                                        Symbol* const destination) const {
        Symbol::copy_symbol_sequence(destination, Symbol::from(this), integrand_offset);
        destination[0].integral.substitution_count += 1;
        destination[0].integral.integrand_offset += 1 + substitution->total_size();
        destination[0].integral.total_size += 1 + substitution->total_size();

        Symbol* current_substitution = destination->child();
        for (size_t i = 0; i < substitution_count; ++i) {
            current_substitution->substitution.sub_substitution_count += 1;
            current_substitution->substitution.total_size += 1 + substitution->total_size();
            current_substitution = current_substitution->substitution.next_substitution();
        }

        current_substitution->substitution = Substitution::create();
        current_substitution->substitution.sub_substitution_count = 0;
        current_substitution->substitution.substitution_idx = substitution_count;
        current_substitution->substitution.total_size =
            destination->total_size() - integrand_offset;
        substitution->copy_to(current_substitution + 1);
    }

    __host__ __device__ void Integral::integrate_by_substitution_with_derivative(
        const Symbol* const substitution, const Symbol* const derivative, Symbol* const destination,
        Symbol* const swap_space) const {
        integrand()->substitute_with_var_with_holes(destination, substitution);
        size_t new_incomplete_integrand_size =
            Symbol::from(destination)->compress_reverse_to(swap_space);
        Symbol::reverse_symbol_sequence(swap_space, new_incomplete_integrand_size);

        size_t old_integrand_size = total_size - integrand_offset;
        size_t new_integrand_size = new_incomplete_integrand_size + 2 + derivative->total_size();
        ssize_t size_diff = new_integrand_size - old_integrand_size;

        copy_substitutions_with_an_additional_one(substitution, destination);

        destination[0].integral.total_size += size_diff;
        Symbol* current_substitution = destination->child();
        for (size_t i = 0; i < destination->integral.substitution_count; ++i) {
            current_substitution->substitution.total_size += size_diff;
            current_substitution = current_substitution->substitution.next_substitution();
        }

        destination->integrand()[0].product = Product::create();
        destination->integrand()[0].product.total_size = new_integrand_size;
        destination->integrand()[0].product.second_arg_offset = 2 + derivative->total_size();
        destination->integrand()[1].reciprocal = Reciprocal::create();
        destination->integrand()[1].reciprocal.total_size = 1 + derivative->total_size();
        derivative->copy_to(destination->integrand() + 2);
        swap_space->copy_to(destination->integrand() + 2 + derivative->total_size());
    }

    __host__ __device__ Substitution* Integral::first_substitution() {
        return &Symbol::from(this)[1].substitution;
    }

    std::vector<Symbol> integral(const std::vector<Symbol>& arg) {
        std::vector<Symbol> integral(arg.size() + 1);
        integral[0].integral = Integral::create();
        integral[0].integral.total_size = 1 + arg[0].total_size();
        integral[0].integral.substitution_count = 0;
        integral[0].integral.integrand_offset = 1;
        std::copy(arg.begin(), arg.end(), integral.begin() + 1);

        return integral;
    }
}
