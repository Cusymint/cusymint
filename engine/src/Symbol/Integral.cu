#include "Integral.cuh"

#include "Substitution.cuh"
#include "Symbol.cuh"
#include <fmt/core.h>

namespace Sym {
    DEFINE_INTO_DESTINATION_OPERATOR(Integral)

    DEFINE_COMPRESS_REVERSE_TO(Integral) {
        size_t new_integrand_size = integrand()->compress_reverse_to(destination);
        size_t new_substitutions_size = 0;

        if (substitution_count > 0) {
            new_substitutions_size = first_substitution()->compress_reverse_substitutions_to(
                destination + new_integrand_size);
        }

        size_t integral_offset = new_integrand_size + new_substitutions_size;

        copy_single_to(destination + integral_offset);
        destination[integral_offset].integral.size = integral_offset + 1;
        destination[integral_offset].integral.integrand_offset = new_substitutions_size + 1;

        return destination[integral_offset].integral.size;
    }

    DEFINE_COMPARE(Integral) {
        return BASE_COMPARE(Integral) &&
               symbol->integral.substitution_count == substitution_count &&
               symbol->integral.integrand_offset == integrand_offset;
    }

    DEFINE_SIMPLIFY_IN_PLACE(Integral) { integrand()->simplify_in_place(help_space); }

    __host__ __device__ void Integral::seal_no_substitutions() { seal_substitutions(0, 0); }

    __host__ __device__ void Integral::seal_single_substitution() {
        seal_substitutions(1, (Symbol::from(this) + integrand_offset)->size());
    }

    __host__ __device__ void Integral::seal_substitutions(const size_t count, const size_t size) {
        integrand_offset = 1 + size;
        substitution_count = count;
    }

    __host__ __device__ void Integral::seal() { size = integrand_offset + integrand()->size(); }

    __host__ __device__ Symbol* Integral::integrand() {
        return Symbol::from(this) + integrand_offset;
    }

    __host__ __device__ const Symbol* Integral::integrand() const {
        return Symbol::from(this) + integrand_offset;
    }

    __host__ __device__ void
    Integral::copy_substitutions_with_an_additional_one(const Symbol* const substitution_expr,
                                                        Symbol* const destination) const {
        Symbol::copy_symbol_sequence(destination, symbol(), integrand_offset);

        Symbol* const new_substitution = destination + integrand_offset;
        Substitution::create(substitution_expr, new_substitution, substitution_count);

        destination->integral.substitution_count += 1;
        destination->integral.integrand_offset += new_substitution->size();
        destination->integral.size += new_substitution->size();
    }

    __host__ __device__ void Integral::copy_without_integrand_to(Symbol* const destination) const {
        Symbol::copy_symbol_sequence(destination, symbol(), 1 + substitutions_size());
    }

    __host__ __device__ void Integral::integrate_by_substitution_with_derivative(
        const Symbol* const substitution, const Symbol* const derivative, Symbol* const destination,
        Symbol* const help_space) const {
        integrand()->substitute_with_var_with_holes(destination, substitution);
        size_t new_incomplete_integrand_size = destination->compress_reverse_to(help_space);
        Symbol::reverse_symbol_sequence(help_space, new_incomplete_integrand_size);

        // Teraz w `help_space` jest docelowa funkcja podcałkowa, ale jeszcze bez mnożenia przez
        // pochodną. W `destination` są niepotrzebne dane.

        const auto old_integrand_size = static_cast<ssize_t>(integrand()->size());
        const auto new_integrand_size =
            static_cast<ssize_t>(new_incomplete_integrand_size + 2 + derivative->size());
        const ssize_t size_diff = new_integrand_size - old_integrand_size;

        copy_substitutions_with_an_additional_one(substitution, destination);

        destination->size() += size_diff;

        Product* const product = destination->integrand() << Product::builder();
        Reciprocal* const reciprocal = product->arg1() << Reciprocal::builder();
        derivative->copy_to(&reciprocal->arg());
        reciprocal->seal();
        product->seal_arg1();
        help_space->copy_to(&product->arg2());
        product->seal();
    }

    __host__ __device__ const Substitution* Integral::first_substitution() const {
        return &Symbol::from(this)->child()->substitution;
    }

    __host__ __device__ Substitution* Integral::first_substitution() {
        return &Symbol::from(this)->child()->substitution;
    }

    __host__ __device__ size_t Integral::substitutions_size() const {
        return size - 1 - integrand()->size();
    };

    std::string Integral::to_string() const {
        std::vector<Symbol> integrand_copy(integrand()->size());
        integrand()->copy_to(integrand_copy.data());

        std::string last_substitution_name;

        if (substitution_count == 0) {
            return fmt::format("∫{}dx", integrand_copy.data()->to_string());
        }

        last_substitution_name = Substitution::nth_substitution_name(substitution_count - 1);
        integrand_copy.data()->substitute_variable_with_nth_substitution_name(substitution_count -
                                                                              1);

        return fmt::format("∫{}d{}, {}", integrand_copy.data()->to_string(), last_substitution_name,
                           first_substitution()->to_string());
    }

    std::string Integral::to_tex() const {
        std::vector<Symbol> integrand_copy(integrand()->size());
        integrand()->copy_to(integrand_copy.data());

        std::string last_substitution_name;

        if (substitution_count == 0) {
            return fmt::format(R"(\int {}\text{{d}}x)", integrand_copy.data()->to_tex());
        }

        last_substitution_name = Substitution::nth_substitution_name(substitution_count - 1);
        integrand_copy.data()->substitute_variable_with_nth_substitution_name(substitution_count -
                                                                              1);

        return fmt::format(R"(\int {}\text{{d}}{},\quad {})", integrand_copy.data()->to_tex(),
                           last_substitution_name, first_substitution()->to_tex());
    }

    std::vector<Symbol> integral(const std::vector<Symbol>& arg) {
        std::vector<Symbol> res(arg.size() + 1);

        Integral* const integral = res.data() << Integral::builder();
        integral->seal_no_substitutions();
        arg.data()->copy_to(integral->integrand());
        integral->seal();

        return res;
    }
}
