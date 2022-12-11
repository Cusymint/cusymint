#include "Integral.cuh"

#include <cstddef>

#include <fmt/core.h>

#include "Macros.cuh"
#include "MetaOperators.cuh"
#include "Substitution.cuh"
#include "Symbol.cuh"

#include "Evaluation/StaticFunctions.cuh"
#include "Utils/Pair.cuh"

namespace Sym {
    DEFINE_INTO_DESTINATION_OPERATOR(Integral)
    DEFINE_IDENTICAL_COMPARE_TO(Integral)
    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(Integral)
    DEFINE_INVALID_DERIVATIVE(Integral)
    DEFINE_INVALID_SEAL_WHOLE(Integral)

    DEFINE_COMPRESS_REVERSE_TO(Integral) {
        size_t new_substitutions_size = 0;
        Symbol* substitution = destination - 1;
        // we assume that substitutions do not need additional size (this is naive imo)
        for (size_t index = substitution_count; index > 0; --index) {
            new_substitutions_size += substitution->size();
            substitution -= substitution->size();
        }
        // now `substitution` points to an integrand
        substitution->size() += substitution->additional_required_size();
        substitution->additional_required_size() = 0;

        const size_t new_integrand_size = substitution->size();

        symbol()->copy_single_to(destination);
        destination->integral.size = new_integrand_size + new_substitutions_size + 1;
        destination->integral.integrand_offset = new_substitutions_size + 1;

        return 1;
    }

    DEFINE_ARE_EQUAL(Integral) {
        return BASE_ARE_EQUAL(Integral) &&
               symbol->integral.substitution_count == substitution_count &&
               symbol->integral.integrand_offset == integrand_offset;
    }

    DEFINE_IS_FUNCTION_OF(Integral) {
        return integrand()->is_function_of(expressions, expression_count);
    }

    DEFINE_PUSH_CHILDREN_ONTO_STACK(Integral) {
        if (substitution_count > 0) {
            stack.push(first_substitution()->symbol());
        }

        stack.push(integrand());
    }

    DEFINE_PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE(Integral) {
        push_children_onto_stack(stack);
        integrand()->additional_required_size() += additional_required_size;
    }

    __host__ __device__ void Integral::seal_no_substitutions() { seal_substitutions(0, 0); }

    __host__ __device__ void Integral::seal_single_substitution() {
        seal_substitutions(1, (symbol() + integrand_offset)->size());
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
    Integral::copy_substitutions_with_an_additional_one(const Symbol& substitution_expr,
                                                        Symbol& destination) const {
        Symbol::copy_symbol_sequence(&destination, symbol(), integrand_offset);

        Symbol* const new_substitution = &destination + integrand_offset;
        Substitution::create(&substitution_expr, new_substitution, substitution_count);

        destination.integral.substitution_count += 1;
        destination.integral.integrand_offset += new_substitution->size();
        destination.integral.size += new_substitution->size();
    }

    __host__ __device__ void Integral::copy_without_integrand_to(Symbol* const destination) const {
        Symbol::copy_symbol_sequence(destination, symbol(), 1 + substitutions_size());
        destination->as<Integral>().size = BUILDER_SIZE;
    }

    __device__ void Integral::integrate_by_substitution_with_derivative(const Symbol& substitution,
                                                                        const Symbol& derivative,
                                                                        Symbol& destination) const {
        const Util::Pair<const Symbol*, const Symbol*> substitution_pairs[] = {
            Util::Pair(&substitution, &Static::identity())};

        integrate_by_substitution_with_derivative(substitution_pairs, 1, derivative, destination);
    }

    __host__ __device__ void Integral::integrate_by_substitution_with_derivative(
        const Util::Pair<const Sym::Symbol*, const Sym::Symbol*>* const patterns,
        const size_t pattern_count, const Symbol& derivative, Symbol& destination) const {

        if constexpr (Consts::DEBUG) {
            if (!patterns[0].second->is(Type::Variable)) {
                Util::crash("The first element of `substitutions` passed to "
                            "`integrate_by_substitution_with_derivative` should be a pair in the "
                            "form of (f(x), x)!");
            }
        }

        copy_substitutions_with_an_additional_one(*patterns[0].first, destination);
        Symbol& destination_integrand = *destination.as<Integral>().integrand();
        Symbol* current_dst = &destination_integrand;

        Mul<Inv<Copy>, Copy>::init(*current_dst, {derivative, *Unknown::create().symbol()});
        current_dst += 2 + derivative.size();

        for (size_t symbol_idx = 0; symbol_idx < integrand()->size(); ++symbol_idx) {
            bool found_match = false;
            for (size_t pattern_idx = 0; pattern_idx < pattern_count; ++pattern_idx) {
                if (!Symbol::are_expressions_equal(integrand()[symbol_idx],
                                                   *patterns[pattern_idx].first)) {
                    continue;
                }

                patterns[pattern_idx].second->copy_to(current_dst);
                current_dst += current_dst->size();
                // -1 because +1 is going to be added by loop control
                symbol_idx += integrand()[symbol_idx].size() - 1;
                found_match = true;
                break;
            }

            if (!found_match) {
                integrand()[symbol_idx].copy_single_to(current_dst);
                current_dst += 1;
            }
        }

        // Sizes and offsets are completely messed up in help_space (but there are no holes), so
        // this is required
        Symbol::seal_whole(destination_integrand, current_dst - &destination_integrand);
        destination.as<Integral>().seal();
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

    std::vector<Symbol> integral(const std::vector<Symbol>& arg,
                                 const std::vector<std::vector<Symbol>>& substitutions) {
        size_t res_size = arg.size() + 1;
        for (const auto& sub : substitutions) {
            res_size += sub.size() + 1;
        }
        std::vector<Symbol> res(res_size);
        Integral* const integral = res.data() << Integral::builder();
        Symbol* current_dst = res.data()->child();
        for (size_t i = 0; i < substitutions.size(); ++i) {
            Substitution::create(substitutions[i].data(), current_dst, i);
            current_dst += current_dst->size();
        }
        integral->seal_substitutions(substitutions.size(), current_dst - res.data()->child());
        arg.data()->copy_to(integral->integrand());
        integral->seal();

        return res;
    }
}
