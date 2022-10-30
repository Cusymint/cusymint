#include "Symbol.cuh"

#include "Symbol/SymbolType.cuh"
#include "Utils/Cuda.cuh"
#include <cstddef>
#include <sys/types.h>

namespace Sym {
    __host__ __device__ void Symbol::copy_symbol_sequence(Symbol* const destination,
                                                          const Symbol* const source,
                                                          size_t symbol_count) {
        Util::copy_mem(destination, source, symbol_count * sizeof(Symbol));
    }

    __host__ __device__ void Symbol::copy_and_reverse_symbol_sequence(Symbol* const destination,
                                                                      const Symbol* const source,
                                                                      size_t symbol_count) {
        for (size_t i = 0; i < symbol_count; ++i) {
            source[symbol_count - i - 1].copy_single_to(destination + i);
        }
    }

    __host__ __device__ bool Symbol::compare_symbol_sequences(const Symbol* sequence1,
                                                              const Symbol* sequence2,
                                                              size_t symbol_count) {
        // Cannot simply use Util::compare_mem because padding can differ
        for (size_t i = 0; i < symbol_count; ++i) {
            if (sequence1[i] != sequence2[i]) {
                return false;
            }
        }

        return true;
    }

    __host__ __device__ void Symbol::swap_symbols(Symbol* const symbol1, Symbol* const symbol2) {
        Util::swap_mem(symbol1, symbol2, sizeof(Symbol));
    }

    __host__ __device__ void Symbol::reverse_symbol_sequence(Symbol* const sequence,
                                                             size_t symbol_count) {
        for (size_t i = 0; i < symbol_count / 2; ++i) {
            swap_symbols(sequence + i, sequence + symbol_count - i - 1);
        }
    }

    __host__ __device__ void Symbol::copy_single_to(Symbol* const destination) const {
        Util::copy_mem(destination, this, sizeof(Symbol));
    }

    __host__ __device__ void Symbol::copy_to(Symbol* const destination) const {
        copy_symbol_sequence(destination, this, size());
    }

    __host__ __device__ bool Symbol::is_constant() const {
        for (size_t i = 0; i < size(); ++i) {
            if (this[i].is(Type::Variable)) {
                return false;
            }
        }

        return true;
    }

    __host__ __device__ ssize_t Symbol::first_var_occurence() const {
        for (ssize_t i = 0; i < size(); ++i) {
            if (this[i].is(Type::Variable)) {
                return i;
            }
        }

        return -1;
    }

    __host__ __device__ bool Symbol::is_function_of(const Symbol* const* const expressions,
                                                    const size_t expression_count) const {
        return VIRTUAL_CALL(*this, is_function_of, expressions, expression_count);
    }

    __host__ __device__ void
    Symbol::substitute_with_var_with_holes(Symbol& destination, const Symbol& expression) const {
        ssize_t first_var_offset = expression.first_var_occurence();
        copy_to(&destination);

        for (size_t i = 0; i < size(); ++i) {
            if (destination[i].is(Type::Variable)) {
                destination[i - first_var_offset].variable = Variable::create();
                i += expression.size() - first_var_offset - 1;
            }
        }
    }

    __host__ __device__ void print_for_debug(Symbol* array, size_t n, const char* msg) {
        printf("%s\n", msg);
        for (size_t i = 0; i < n; ++i) {
            printf("\t%s%s\t%2lu+%lu\n", array[i].unknown.to_be_copied ? "+" : "-",
                   type_name(array[i].type()), array[i].size(),
                   array[i].unknown.additional_required_size);
        }
    }

    __host__ __device__ size_t Symbol::compress_reverse_to(Symbol* const destination) /*const*/ {
        mark_to_be_copied_and_propagate_additional_size(destination);

        //print_for_debug(this, size(), "After mark():");

        Symbol* compressed_reversed_destination = destination;
        for (ssize_t i = size() - 1; i >= 0; --i) {
            if (at(i)->unknown.to_be_copied) {
                at(i)->unknown.to_be_copied = false;

                const size_t new_size =
                    // at(i)->compress_reverse_to(compressed_reversed_destination);
                    VIRTUAL_CALL(*at(i), compress_reverse_to, compressed_reversed_destination);
                compressed_reversed_destination += new_size;
            }
        }

        // print_for_debug(destination, compressed_reversed_destination-destination, "After
        // compress():");
        return compressed_reversed_destination - destination;
        // return VIRTUAL_CALL(*this, compress_reverse_to, destination);
    }

    __host__ __device__ void
    Symbol::mark_to_be_copied_and_propagate_additional_size(Symbol* const help_space) {
        Symbol** stack = reinterpret_cast<Symbol**>(help_space);
        size_t stack_index = 0;

        stack[stack_index++] = this;

        while (stack_index > 0) {
            Symbol* sym = stack[--stack_index];

            sym->unknown.to_be_copied = true;

            switch (sym->unknown.type) {
            case Type::Symbol:
                Util::crash("Trying to access children on a pure Symbol");
                break;
            case Type::Variable:
                break;
            case Type::NumericConstant:
                break;
            case Type::KnownConstant:
                break;
            case Type::UnknownConstant:
                break;
            case Type::ExpanderPlaceholder:
                break;
            case Type::SubexpressionCandidate:
                stack[stack_index++] = &sym->subexpression_candidate.arg();
                sym->subexpression_candidate.arg().unknown.additional_required_size +=
                    sym->unknown.additional_required_size;

                // sym->unknown.additional_required_size = 0;
                break;
            case Type::SubexpressionVacancy:
                break;
            case Type::Integral:
                stack[stack_index++] = sym->integral.integrand();
                if (sym->integral.substitution_count > 0)
                    stack[stack_index++] = sym->integral.first_substitution()->symbol();
                sym->integral.integrand()->unknown.additional_required_size +=
                    sym->unknown.additional_required_size;

                // sym->unknown.additional_required_size = 0;
                break;
            case Type::Solution:
                stack[stack_index++] = sym->solution.expression();
                if (sym->solution.substitution_count > 0)
                    stack[stack_index++] = sym->solution.first_substitution()->symbol();
                sym->solution.expression()->unknown.additional_required_size +=
                    sym->unknown.additional_required_size;

                // sym->unknown.additional_required_size = 0;
                break;
            case Type::Substitution:
                stack[stack_index++] = sym->substitution.expression();
                if (!sym->substitution.is_last_substitution())
                    stack[stack_index++] = sym->substitution.next_substitution()->symbol();
                sym->substitution.expression()->unknown.additional_required_size +=
                    sym->unknown.additional_required_size;

                // sym->unknown.additional_required_size = 0;
                break;
            case Type::Addition:
                stack[stack_index++] = &sym->addition.arg1();
                stack[stack_index++] = &sym->addition.arg2();
                sym->addition.arg2().unknown.additional_required_size +=
                    sym->unknown.additional_required_size;

                // sym->unknown.additional_required_size = 0;
                break;
            case Type::Negation:
                stack[stack_index++] = &sym->negation.arg();
                sym->negation.arg().unknown.additional_required_size +=
                    sym->unknown.additional_required_size;

                // sym->unknown.additional_required_size = 0;
                break;
            case Type::Product:
                stack[stack_index++] = &sym->product.arg1();
                stack[stack_index++] = &sym->product.arg2();
                sym->product.arg2().unknown.additional_required_size +=
                    sym->unknown.additional_required_size;

                // sym->unknown.additional_required_size = 0;
                break;
            case Type::Reciprocal:
                stack[stack_index++] = &sym->reciprocal.arg();
                sym->reciprocal.arg().unknown.additional_required_size +=
                    sym->unknown.additional_required_size;

                // sym->unknown.additional_required_size = 0;
                break;
            case Type::Power:
                stack[stack_index++] = &sym->power.arg1();
                stack[stack_index++] = &sym->power.arg2();
                sym->power.arg2().unknown.additional_required_size +=
                    sym->unknown.additional_required_size;

                // sym->unknown.additional_required_size = 0;
                break;
            case Type::Sine:
                stack[stack_index++] = &sym->sine.arg();
                sym->sine.arg().unknown.additional_required_size +=
                    sym->unknown.additional_required_size;

                // sym->unknown.additional_required_size = 0;
                break;
            case Type::Cosine:
                stack[stack_index++] = &sym->cosine.arg();
                sym->cosine.arg().unknown.additional_required_size +=
                    sym->unknown.additional_required_size;

                // sym->unknown.additional_required_size = 0;
                break;
            case Type::Tangent:
                stack[stack_index++] = &sym->tangent.arg();
                sym->tangent.arg().unknown.additional_required_size +=
                    sym->unknown.additional_required_size;

                // sym->unknown.additional_required_size = 0;
                break;
            case Type::Cotangent:
                stack[stack_index++] = &sym->cotangent.arg();
                sym->cotangent.arg().unknown.additional_required_size +=
                    sym->unknown.additional_required_size;

                // sym->unknown.additional_required_size = 0;
                break;
            case Type::Arcsine:
                stack[stack_index++] = &sym->arcsine.arg();
                sym->arcsine.arg().unknown.additional_required_size +=
                    sym->unknown.additional_required_size;

                // sym->unknown.additional_required_size = 0;
                break;
            case Type::Arccosine:
                stack[stack_index++] = &sym->arccosine.arg();
                sym->arccosine.arg().unknown.additional_required_size +=
                    sym->unknown.additional_required_size;

                // sym->unknown.additional_required_size = 0;
                break;
            case Type::Arctangent:
                stack[stack_index++] = &sym->arctangent.arg();
                sym->arctangent.arg().unknown.additional_required_size +=
                    sym->unknown.additional_required_size;

                // sym->unknown.additional_required_size = 0;
                break;
            case Type::Arccotangent:
                stack[stack_index++] = &sym->arccotangent.arg();
                sym->arccotangent.arg().unknown.additional_required_size +=
                    sym->unknown.additional_required_size;

                // sym->unknown.additional_required_size = 0;
                break;
            case Type::Logarithm:
                stack[stack_index++] = &sym->logarithm.arg();
                sym->logarithm.arg().unknown.additional_required_size +=
                    sym->unknown.additional_required_size;

                // sym->unknown.additional_required_size = 0;
                break;
            case Type::Unknown:
                break;
            default:
                Util::crash("Trying to access children of invalid type");
                break;
            }
        }
    }

    __host__ __device__ void Symbol::simplify(Symbol* const help_space) {
        bool success = true;

        //int _debug_i = 0;

        do {
            success = true;
            //print_for_debug(this, size(), "Before simplify_in_place():");

            for (ssize_t i = size() - 1; i >= 0; --i) {
                success = at(i)->simplify_in_place(help_space) && success;
            }

            //print_for_debug(this, size(), "After simplify_in_place():");

            // mark_to_be_copied_and_propagate_additional_size(help_space);

            // //print_for_debug(this, size(), "After simplify_in_place():");

            // Symbol* compressed_reversed_destination = help_space;
            // for (ssize_t i = size() - 1; i >= 0; --i) {
            //     if (at(i)->unknown.to_be_copied) {
            //         at(i)->unknown.to_be_copied = false;

            //         const size_t new_size =
            //             at(i)->compress_reverse_to(compressed_reversed_destination);
            //         compressed_reversed_destination += new_size;
            //     }
            // }
            const size_t new_size = compress_reverse_to(help_space);

            // print_for_debug(help_space, new_size, "After compress_reverse_to():");
            // printf("Symbols copied: %ld\n", new_size);

            copy_and_reverse_symbol_sequence(this, help_space, new_size);

            //print_for_debug(this, size(), "After iteration():");

            // if (_debug_i++ > 2) {
            //     Util::crash("Loop terminated.");
            // }
        } while (!success);
    }

    __host__ __device__ bool Symbol::simplify_in_place(Symbol* const help_space) {
        return VIRTUAL_CALL(*this, simplify_in_place, help_space);
    }

    void Symbol::substitute_variable_with(const Symbol symbol) {
        for (size_t i = 0; i < size(); ++i) {
            if (this[i].is(Type::Variable)) {
                this[i] = symbol;
            }
        }
    }

    void Symbol::substitute_variable_with_nth_substitution_name(const size_t n) {
        std::string substitution_name = Substitution::nth_substitution_name(n);

        Symbol substitute{};
        substitute.unknown_constant = UnknownConstant::create(substitution_name.c_str());
        substitute_variable_with(substitute);
    }

    __host__ __device__ bool Symbol::compare_trees(const Symbol* const expr1,
                                                   const Symbol* const expr2) {
        if (expr1->size() != expr2->size()) {
            return false;
        }

        return compare_symbol_sequences(expr1, expr2, expr1->size());
    }

    std::string Symbol::to_string() const { return VIRTUAL_CALL(*this, to_string); }

    std::string Symbol::to_tex() const { return VIRTUAL_CALL(*this, to_tex); }

    __host__ __device__ bool operator==(const Symbol& sym1, const Symbol& sym2) {
        return VIRTUAL_CALL(sym1, compare, &sym2);
    }

    __host__ __device__ bool operator!=(const Symbol& sym1, const Symbol& sym2) {
        return !(sym1 == sym2);
    }
};
