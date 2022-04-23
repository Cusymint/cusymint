#ifndef INTEGRAL_EXPRESSION_CUH
#define INTEGRAL_EXPRESSION_CUH

#include "symbol.cuh"

namespace Sym {
    struct IntegralExpression {
        static constexpr size_t MAX_SYMBOL_COUNT = 64;
        static constexpr size_t MAX_SUBSTITUTION_COUNT = 4;
        static const char* const SUBSTITUTION_NAMES[MAX_SUBSTITUTION_COUNT];
        static constexpr size_t MAX_SUBSTITUTION_SIZE = 32;

        static IntegralExpression from_symbols(const Symbol* const symbols);
        static IntegralExpression from_symbols(const std::vector<Symbol>& symbols);

        Symbol symbols[MAX_SYMBOL_COUNT];
        bool solved;

        size_t substitution_count;
        Symbol substitutions[MAX_SUBSTITUTION_SIZE][MAX_SUBSTITUTION_COUNT];

        std::string to_string();
    };
}

#endif
