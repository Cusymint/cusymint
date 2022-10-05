#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

#include <vector>

#include "Macros.cuh"

namespace Sym {
    enum class KnownConstantValue { Unknown, Pi, E };

    DECLARE_SYMBOL(NumericConstant, true)
    double value;
    __host__ __device__ NumericConstant static with_value(double value);
    DEFINE_TO_STRING(std::to_string(value));
    DEFINE_TO_TEX(std::to_string(value));
    END_DECLARE_SYMBOL(NumericConstant)

    DECLARE_SYMBOL(KnownConstant, true)
    KnownConstantValue value;
    __host__ __device__ KnownConstant static with_value(KnownConstantValue value);
    std::string to_string() const;
    std::string to_tex() const;
    END_DECLARE_SYMBOL(KnownConstant)

    DECLARE_SYMBOL(UnknownConstant, true)
    static constexpr size_t NAME_LEN = 8;
    char name[NAME_LEN];
    static UnknownConstant create(const char* const name);
    DEFINE_TO_STRING(name)
    DEFINE_TO_TEX(name)
    END_DECLARE_SYMBOL(UnknownConstant)

    std::vector<Symbol> known_constant(KnownConstantValue value);
    std::vector<Symbol> e();
    std::vector<Symbol> pi();
    std::vector<Symbol> cnst(const char name[UnknownConstant::NAME_LEN]);
    std::vector<Symbol> num(double value);
}

#endif
