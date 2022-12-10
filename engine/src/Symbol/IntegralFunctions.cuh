#ifndef INTEGRAL_FUNCTIONS_CUH
#define INTEGRAL_FUNCTIONS_CUH

#include <vector>

#include "Macros.cuh"

namespace Sym {
    DECLARE_SYMBOL(SineIntegral, false)
    ONE_ARGUMENT_OP_SYMBOL

    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::string to_tex() const;
    END_DECLARE_SYMBOL(SineIntegral)

    DECLARE_SYMBOL(CosineIntegral, false)
    ONE_ARGUMENT_OP_SYMBOL

    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::string to_tex() const;
    END_DECLARE_SYMBOL(CosineIntegral)

    DECLARE_SYMBOL(ExponentialIntegral, false)
    ONE_ARGUMENT_OP_SYMBOL

    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::string to_tex() const;
    END_DECLARE_SYMBOL(ExponentialIntegral)

    DECLARE_SYMBOL(LogarithmicIntegral, false)
    ONE_ARGUMENT_OP_SYMBOL

    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::string to_tex() const;
    END_DECLARE_SYMBOL(LogarithmicIntegral)

    std::vector<Symbol> Si(const std::vector<Symbol>& arg);
    std::vector<Symbol> Ci(const std::vector<Symbol>& arg);
    std::vector<Symbol> Ei(const std::vector<Symbol>& arg);
    std::vector<Symbol> li(const std::vector<Symbol>& arg);
}

#endif