#include "Constants.cuh"

#include <fmt/core.h>
#include <stdexcept>
#include <string>

#include "Symbol.cuh"
#include "Utils/Cuda.cuh"

namespace Sym {
    DEFINE_SIMPLE_COMPRESS_REVERSE_TO(NumericConstant);
    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(NumericConstant);

    DEFINE_SIMPLE_COMPRESS_REVERSE_TO(KnownConstant);
    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(KnownConstant);

    DEFINE_SIMPLE_COMPRESS_REVERSE_TO(UnknownConstant);
    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(UnknownConstant);

    DEFINE_COMPARE(NumericConstant) {
        return BASE_COMPARE(NumericConstant) && symbol->numeric_constant.value == value;
    }

    DEFINE_COMPARE(UnknownConstant) {
        return BASE_COMPARE(UnknownConstant) &&
               Util::compare_mem(symbol->unknown_constant.name, name, NAME_LEN);
    }

    DEFINE_COMPARE(KnownConstant) {
        return BASE_COMPARE(KnownConstant) && symbol->known_constant.value == value;
    }

    __host__ __device__ NumericConstant NumericConstant::with_value(double value) {
        NumericConstant constant = NumericConstant::create();
        constant.value = value;
        return constant;
    }

    __host__ __device__ KnownConstant KnownConstant::with_value(KnownConstantValue value) {
        KnownConstant constant = KnownConstant::create();
        constant.value = value;
        return constant;
    }

    std::string KnownConstant::to_string() const {
        switch (value) {
        case KnownConstantValue::E:
            return "e";
        case KnownConstantValue::Pi:
            return "π";
        case KnownConstantValue::Unknown:
        default:
            return "<Undefined known constant>";
        }
    }

    std::string KnownConstant::to_tex() const {
        switch (value) {
        case KnownConstantValue::E:
            return "\\text{e}";
        case KnownConstantValue::Pi:
            return "\\pi";
        case KnownConstantValue::Unknown:
        default:
            return "?";
        }
    }

    std::string NumericConstant::to_tex() const {
        if (value < 0) {
            return fmt::format("\\left( {} \\right)", value);
        }
        return std::to_string(value);
    }

    UnknownConstant UnknownConstant::create(const char* const name) {
        UnknownConstant unknown_constant = UnknownConstant::create();
        if (strlen(name) + 1 > NAME_LEN) {
            throw std::invalid_argument("UnknownConstant created with name longer than NAME_LEN");
        }

        strcpy(unknown_constant.name, name);
        return unknown_constant;
    }

    std::vector<Symbol> known_constant(KnownConstantValue value) {
        std::vector<Symbol> symbol_vec(1);
        symbol_vec[0].known_constant = KnownConstant::create();
        symbol_vec[0].known_constant.value = value;
        return symbol_vec;
    }

    std::vector<Symbol> e() { return known_constant(KnownConstantValue::E); }

    std::vector<Symbol> pi() { return known_constant(KnownConstantValue::Pi); }

    std::vector<Symbol> cnst(const char name[UnknownConstant::NAME_LEN]) {
        std::vector<Symbol> symbol_vec(1);
        symbol_vec[0].unknown_constant = UnknownConstant::create();
        std::copy(name, name + UnknownConstant::NAME_LEN, symbol_vec[0].unknown_constant.name);
        return symbol_vec;
    }

    std::vector<Symbol> num(double value) {
        std::vector<Symbol> symbol_vec(1);
        symbol_vec[0].numeric_constant = NumericConstant::create();
        symbol_vec[0].numeric_constant.value = value;
        return symbol_vec;
    }
}
