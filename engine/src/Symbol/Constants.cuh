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
    [[nodiscard]] std::string to_tex() const;
    DEFINE_IS_POLYNOMIAL(0)
    DEFINE_IS_MONOMIAL(value)
    END_DECLARE_SYMBOL(NumericConstant)

    DECLARE_SYMBOL(KnownConstant, true)
    KnownConstantValue value;
    __host__ __device__ KnownConstant static with_value(KnownConstantValue value);
    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::string to_tex() const;
    DEFINE_IS_NOT_POLYNOMIAL // TODO: operations on non-numeric constants
    END_DECLARE_SYMBOL(KnownConstant)

    DECLARE_SYMBOL(UnknownConstant, true)
    static constexpr size_t NAME_LEN = 8;
    char name[NAME_LEN];
    static UnknownConstant create(const char* const name);
    DEFINE_TO_STRING(name)
    DEFINE_TO_TEX(name)
    DEFINE_IS_NOT_POLYNOMIAL // TODO: operations on non-numeric constants
    END_DECLARE_SYMBOL(UnknownConstant)

    std::vector<Symbol> known_constant(KnownConstantValue value);
    std::vector<Symbol> e();
    std::vector<Symbol> pi();
    std::vector<Symbol> cnst(const char name[UnknownConstant::NAME_LEN]);
    std::vector<Symbol> num(double value);
}

#endif
