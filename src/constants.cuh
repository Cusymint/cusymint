#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

#include <vector>

#include "symbol_defs.cuh"

namespace Sym {
    enum class KnownConstantValue { Unknown, Pi, E };

    DECLARE_SYMBOL(NumericConstant, true)
    double value;
    DEFINE_TO_STRING(std::to_string(value));
    END_DECLARE_SYMBOL(NumericConstant)

    DECLARE_SYMBOL(KnownConstant, true)
    KnownConstantValue value;
    std::string to_string() const;
    END_DECLARE_SYMBOL(KnownConstant)

    DECLARE_SYMBOL(UnknownConstant, true)
    static constexpr size_t NAME_LEN = 8;
    char name[NAME_LEN];
    static UnknownConstant create(const char* const name);
    DEFINE_TO_STRING(name)
    END_DECLARE_SYMBOL(UnknownConstant)

    std::vector<Symbol> known_constant(KnownConstantValue value);
    std::vector<Symbol> e();
    std::vector<Symbol> pi();
    std::vector<Symbol> cnst(const char name[UnknownConstant::NAME_LEN]);
    std::vector<Symbol> num(double value);
}

#endif
