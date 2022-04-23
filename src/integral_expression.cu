#include "integral_expression.cuh"

namespace Sym {
    const char* const IntegralExpression::SUBSTITUTION_NAMES[MAX_SUBSTITUTION_COUNT] = {"u", "v",
                                                                                        "w", "t"};

    IntegralExpression IntegralExpression::from_symbols(const Symbol* const symbols) {
        IntegralExpression expression = {.solved = false, .substitution_count = 0};

        memcpy(expression.symbols, symbols, sizeof(Symbol) * symbols->unknown.total_size);

        return expression;
    }

    IntegralExpression IntegralExpression::from_symbols(const std::vector<Symbol>& symbols) {
        return from_symbols(symbols.data());
    }

    std::string IntegralExpression::to_string() {
        Symbol substitution_symbols[MAX_SYMBOL_COUNT];
        memcpy(substitution_symbols, symbols, sizeof(Symbol) * MAX_SYMBOL_COUNT);
        std::string str;

        for (int i = substitution_count - 1; i >= 0; --i) {
            Symbol renamed_variable;
            renamed_variable.unknown_constant = UnknownConstant::create(SUBSTITUTION_NAMES[i]);
            substitution_symbols->substitute_variable_with(renamed_variable);

            str += substitution_symbols->to_string() + ", where " + SUBSTITUTION_NAMES[i] + " = ";

            memcpy(substitution_symbols, substitutions[i], MAX_SUBSTITUTION_SIZE * sizeof(Symbol));
        }

        str += substitution_symbols[0].to_string();

        return str;
    }
}
