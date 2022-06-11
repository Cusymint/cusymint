
#include "addition.cuh"

#include "cuda_utils.cuh"
#include "symbol.cuh"

namespace Sym {
    DEFINE_TWO_ARGUMENT_OP_FUNCTIONS(Addition)
    DEFINE_SIMPLE_ONE_ARGUMETN_OP_COMPARE(Addition)
    DEFINE_TWO_ARGUMENT_OP_COMPRESS_REVERSE_TO(Addition)

    DEFINE_SIMPLIFY_IN_PLACE(Addition) {
        arg1().simplify_in_place(help_space);
        arg2().simplify_in_place(help_space);

        simplify_structure(help_space);
    }

    __host__ __device__ void Addition::simplify_structure(Symbol* const help_space) {
        if (!arg2().is(Type::Addition)) {
            return;
        }

        if (!arg1().is(Type::Addition)) {
            swap_args(help_space);
            return;
        }

        Symbol* const last_left = &arg1().addition.last_in_tree()->arg1();

        Symbol* const last_left_copy = help_space;
        last_left->copy_to(last_left_copy);
        last_left->expander_placeholder = ExpanderPlaceholder::with_size(arg2().size());

        Symbol* const right_copy = help_space + last_left_copy->size();
        arg2().copy_to(right_copy);
        arg2().expander_placeholder = ExpanderPlaceholder::with_size(last_left_copy->size());

        Symbol* const resized_reversed_this = right_copy + right_copy->size();
        compress_reverse_to(resized_reversed_this);

        // Zmiany w strukturze nie zmieniają całkowitego rozmiaru `this`
        Symbol::copy_and_reverse_symbol_sequence(this_symbol(), resized_reversed_this, size);
        right_copy->copy_to(last_left);
        last_left_copy->copy_to(&arg2());
    }

    __host__ __device__ const Addition* Addition::last_in_tree() const {
        if (arg1().is(Type::Addition)) {
            return arg1().addition.last_in_tree();
        }

        return this;
    }

    __host__ __device__ Addition* Addition::last_in_tree() {
        return const_cast<Addition*>(const_cast<const Addition*>(this)->last_in_tree());
    }

    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(Negation)
    DEFINE_SIMPLE_ONE_ARGUMETN_OP_COMPARE(Negation)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Negation)

    DEFINE_SIMPLIFY_IN_PLACE(Negation) { arg().simplify_in_place(help_space); }

    std::string Addition::to_string() const {
        const Symbol* const sym = Symbol::from(this);
        if (arg2().is(Type::Negation)) {
            return "(" + arg1().to_string() + "-" + arg2().negation.arg().to_string() + ")";
        }
        else if (arg2().is(Type::NumericConstant) && arg2().numeric_constant.value < 0.0) {
            return "(" + arg1().to_string() + "-" + std::to_string(-arg2().numeric_constant.value) +
                   ")";
        }
        else {
            return "(" + arg1().to_string() + "+" + arg2().to_string() + ")";
        }
    }

    std::string Negation::to_string() const { return "-" + arg().to_string(); }

    std::vector<Symbol> operator+(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs) {
        std::vector<Symbol> res(lhs.size() + rhs.size() + 1);
        Addition::create(lhs.data(), rhs.data(), res.data());
        return res;
    }

    std::vector<Symbol> operator-(const std::vector<Symbol>& arg) {
        std::vector<Symbol> res(arg.size() + 1);
        Negation::create(arg.data(), res.data());
        return res;
    }

    std::vector<Symbol> operator-(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs) {
        return lhs + (-rhs);
    }

}
