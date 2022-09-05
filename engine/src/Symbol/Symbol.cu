#include "Symbol.cuh"

#include "Utils/Cuda.cuh"

namespace Sym {
    __host__ __device__ void Symbol::copy_symbol_sequence(Symbol* const dst,
                                                          const Symbol* const src, size_t n) {
        Util::copy_mem(dst, src, n * sizeof(Symbol));
    }

    __host__ __device__ void
    Symbol::copy_and_reverse_symbol_sequence(Symbol* const dst, const Symbol* const src, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            src[n - i - 1].copy_single_to(dst + i);
        }
    }

    __host__ __device__ bool Symbol::compare_symbol_sequences(const Symbol* seq1,
                                                              const Symbol* seq2, size_t n) {
        // Cannot simply use Util::compare_mem because padding can differ
        for (size_t i = 0; i < n; ++i) {
            if (seq1[i] != seq2[i]) {
                return false;
            }
        }

        return true;
    }

    __host__ __device__ void Symbol::swap_symbols(Symbol* const symbol1, Symbol* const symbol2) {
        Util::swap_mem(symbol1, symbol2, sizeof(Symbol));
    }

    __host__ __device__ void Symbol::reverse_symbol_sequence(Symbol* const seq, size_t n) {
        for (size_t i = 0; i < n / 2; ++i) {
            swap_symbols(seq + i, seq + n - i - 1);
        }
    }

    __host__ __device__ void Symbol::copy_single_to(Symbol* const dst) const {
        VIRTUAL_CALL(*this, copy_single_to, dst);
    }

    __host__ __device__ void Symbol::copy_to(Symbol* const dst) const {
        Util::copy_mem(dst, this, size() * sizeof(Symbol));
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

    __host__ __device__ bool Symbol::is_function_of(Symbol* expression) const {
        if (is_constant()) {
            return false;
        }

        ssize_t first_var_offset = expression->first_var_occurence();

        for (size_t i = 0; i < size(); ++i) {
            if (this[i].is(Type::Variable)) {
                // First variable in `this` appears earlier than first variable in `expression` or
                // `this` is not long enough to contain another occurence of `expression`
                if (i < first_var_offset || size() - i < expression->size() - first_var_offset) {
                    return false;
                }

                if (!compare_symbol_sequences(this + i - first_var_offset, expression,
                                              expression->size())) {
                    return false;
                }

                i += expression->size() - first_var_offset - 1;
            }
        }

        return true;
    }

    __host__ __device__ void
    Symbol::substitute_with_var_with_holes(Symbol* const destination,
                                           const Symbol* const expression) const {
        ssize_t first_var_offset = expression->first_var_occurence();
        copy_to(destination);

        for (size_t i = 0; i < size(); ++i) {
            if (destination[i].is(Type::Variable)) {
                destination[i - first_var_offset].variable = Variable::create();
                i += expression->size() - first_var_offset - 1;
            }
        }
    }

    __host__ __device__ size_t Symbol::compress_reverse_to(Symbol* const destination) const {
        return VIRTUAL_CALL(*this, compress_reverse_to, destination);
    }

    __host__ __device__ void Symbol::simplify(Symbol* const help_space) {
        simplify_in_place(help_space);
        const size_t new_size = compress_reverse_to(help_space);
        copy_and_reverse_symbol_sequence(this, help_space, new_size);
    }

    __host__ __device__ void Symbol::simplify_in_place(Symbol* const help_space) {
        VIRTUAL_CALL(*this, simplify_in_place, help_space);
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

    __host__ __device__ bool operator==(const Symbol& sym1, const Symbol& sym2) {
        return VIRTUAL_CALL(sym1, compare, &sym2);
    }

    __host__ __device__ bool operator!=(const Symbol& sym1, const Symbol& sym2) {
        return !(sym1 == sym2);
    }
};
