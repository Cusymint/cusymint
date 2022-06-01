
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
