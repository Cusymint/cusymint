#include "Symbol.cuh"

#include "Utils/Cuda.cuh"

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

    __host__ __device__ size_t Symbol::compress_reverse_to(Symbol* const destination) const {
        return VIRTUAL_CALL(*this, compress_reverse_to, destination);
    }

    __host__ __device__ void print_sequence_for_debug(Symbol* symbol) {
        printf("SYMBOLS:\n");
        if (symbol->size()==0)
        {
            printf("\t<no symbols>\n");
            return;
        }
        for (int i=0;i<symbol->size();++i) {
            printf("\t%s\n",type_name(symbol[i].type()));
        }
    }

    __host__ __device__ void Symbol::simplify(Symbol* const help_space) {
        //printf("1: ");
        //print_sequence_for_debug(this);
        simplify_in_place(help_space);
        
        //printf("2: ");print_sequence_for_debug(this);
        const size_t new_size = compress_reverse_to(help_space);
        copy_and_reverse_symbol_sequence(this, help_space, new_size);
        //printf("3: ");
        //print_sequence_for_debug(this);
    }

    __host__ __device__ void Symbol::simplify_in_place(Symbol* const help_space) {
        //printf("Simplifying ");
        //print_sequence_for_debug(this);
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

    std::string Symbol::to_tex() const { return VIRTUAL_CALL(*this, to_tex); }

    __host__ __device__ bool operator==(const Symbol& sym1, const Symbol& sym2) {
        return VIRTUAL_CALL(sym1, compare, &sym2);
    }

    __host__ __device__ bool operator!=(const Symbol& sym1, const Symbol& sym2) {
        return !(sym1 == sym2);
    }
};
