#include "solution.cuh"

#include "symbol.cuh"

namespace Sym {
    std::string Solution::to_string() {
        std::string substitutions_str;

        Symbol* symbols = Symbol::from(this) + symbols_offset;
        std::vector<Symbol> subst_symbols(symbols->unknown.total_size);
        std::copy(symbols, symbols + symbols->unknown.total_size, subst_symbols.data());

        if (substitution_count != 0) {
            std::string last_substitution_name =
                Substitution::nth_substitution_name(substitution_count - 1);
            Symbol* first_substitution = Symbol::from(this) + 1;
            substitutions_str = ", " + first_substitution->to_string();

            Symbol substitute;
            substitute.unknown_constant = UnknownConstant::create(last_substitution_name.c_str());
            subst_symbols[0].substitute_variable_with(substitute);
        }

        return "Solution: " + subst_symbols[0].to_string() + substitutions_str;
    }

    std::vector<Symbol> solution(const std::vector<Symbol>& arg) {
        std::vector<Symbol> solution(arg.size() + 1);
        solution[0].solution = Solution::create();
        solution[0].solution.total_size = 1 + arg[0].unknown.total_size;
        solution[0].solution.substitution_count = 0;
        solution[0].solution.symbols_offset = 1;
        std::copy(arg.begin(), arg.end(), solution.begin() + 1);

        return solution;
    }
}
