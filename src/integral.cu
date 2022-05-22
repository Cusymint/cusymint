#include "integral.cuh"

#include "substitution.cuh"
#include "symbol.cuh"

namespace Sym {
    DEFINE_UNSUPPORTED_COMPRESS_REVERSE_TO(Integral)
    DEFINE_INTO_DESTINATION_OPERATOR(Integral)

    DEFINE_COMPARE(Integral) {
        return BASE_COMPARE(Integral) &&
               symbol->integral.substitution_count == substitution_count &&
               symbol->integral.integrand_offset == integrand_offset;
    }

    __host__ __device__ void Integral::seal_no_substitutions() { seal_substitutions(0, 0); }

    __host__ __device__ void Integral::seal_single_substitution() {
        seal_substitutions(1, (Symbol::from(this) + integrand_offset)->size());
    }

    __host__ __device__ void Integral::seal_substitutions(const size_t count, const size_t size) {
        integrand_offset = 1 + size;
        substitution_count = count;
    }

    __host__ __device__ void Integral::seal() {
        size = integrand_offset + integrand()->size();
    }

    __host__ __device__ Symbol* Integral::integrand() {
        return Symbol::from(this) + integrand_offset;
    }

    __host__ __device__ const Symbol* Integral::integrand() const {
        return Symbol::from(this) + integrand_offset;
    }

    __host__ __device__ void
    Integral::copy_substitutions_with_an_additional_one(const Symbol* const substitution_expr,
                                                        Symbol* const destination) const {
        Symbol::copy_symbol_sequence(destination, Symbol::from(this), integrand_offset);

        Symbol* const new_substitution = destination + integrand_offset;
        Substitution::create(substitution_expr, new_substitution, substitution_count);

        destination->integral.substitution_count += 1;
        destination->integral.integrand_offset += new_substitution->size();
        destination->integral.size += new_substitution->size();
    }

    __host__ __device__ void Integral::integrate_by_substitution_with_derivative(
        const Symbol* const substitution, const Symbol* const derivative, Symbol* const destination,
        Symbol* const swap_space) const {
        integrand()->substitute_with_var_with_holes(destination, substitution);
        size_t new_incomplete_integrand_size = destination->compress_reverse_to(swap_space);
        Symbol::reverse_symbol_sequence(swap_space, new_incomplete_integrand_size);

        // Teraz w `swap_space` jest docelowa funkcja podcałkowa, ale jeszcze bez mnożenia przez
        // pochodną. W `destination` są już niepotrzebne dane.

        size_t old_integrand_size = integrand()->size();
        size_t new_integrand_size = new_incomplete_integrand_size + 2 + derivative->size();
        ssize_t size_diff = new_integrand_size - old_integrand_size;

        copy_substitutions_with_an_additional_one(substitution, destination);

        destination->size() += size_diff;

        Product* const product = destination->integrand() << Product::builder();
        Reciprocal* const reciprocal = product->arg1() << Reciprocal::builder();
        derivative->copy_to(&reciprocal->arg());
        reciprocal->seal();
        product->seal_arg1();
        swap_space->copy_to(&product->arg2());
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
        std::string substitutions_str;
        if (substitution_count == 0) {
            last_substitution_name = "x";
        }
        else {
            substitutions_str = ", " + first_substitution()->to_string();
            last_substitution_name = Substitution::nth_substitution_name(substitution_count - 1);
            integrand_copy.data()->substitute_variable_with_nth_substitution_name(
                substitution_count - 1);
        }

        return "∫" + integrand_copy.data()->to_string() + "d" + last_substitution_name +
               substitutions_str;
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
