#include "Constants.cuh"

#include <cstring>
#include <fmt/core.h>
#include <stdexcept>
#include <string>

#include "Symbol.cuh"
#include "Symbol/Macros.cuh"
#include "Utils/Cuda.cuh"

namespace Sym {
    DEFINE_SIMPLE_COMPRESS_REVERSE_TO(NumericConstant);
    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(NumericConstant);
    DEFINE_IS_FUNCTION_OF(NumericConstant) { return true; } // NOLINT
    DEFINE_NO_OP_PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE(NumericConstant)

    DEFINE_SIMPLE_COMPRESS_REVERSE_TO(KnownConstant);
    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(KnownConstant);
    DEFINE_IS_FUNCTION_OF(KnownConstant) { return true; } // NOLINT
    DEFINE_NO_OP_PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE(KnownConstant)

    DEFINE_SIMPLE_COMPRESS_REVERSE_TO(UnknownConstant);
    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(UnknownConstant);
    DEFINE_IS_FUNCTION_OF(UnknownConstant) { return true; } // NOLINT
    DEFINE_NO_OP_PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE(UnknownConstant)

    DEFINE_ARE_EQUAL(NumericConstant) {
        return BASE_ARE_EQUAL(NumericConstant) && symbol->numeric_constant.value == value;
    }

    DEFINE_COMPARE_TO(NumericConstant) {
        return Util::compare(value, other.as<NumericConstant>().value);
    }

    DEFINE_ARE_EQUAL(UnknownConstant) {
        return BASE_ARE_EQUAL(UnknownConstant) &&
               Util::compare_str(symbol->unknown_constant.name, name, NAME_LEN);
    }

    DEFINE_COMPARE_TO(UnknownConstant) {
        const auto& other_name = other.as<UnknownConstant>().name;
        for (size_t i = 0; i < NAME_LEN; ++i) {
            if (name[i] == '\0' && other_name[i] == '\0') {
                break;
            }

            // This generalizes to the case when both strings are of different length. If name[i] ==
            // '\0' and other.name[i] does not, then this condition will be true. Similarly in the
            // case when the first string is longer.
            if (name[i] < other_name[i]) {
                return Util::Order::Less;
            }

            if (name[i] > other_name[i]) {
                return Util::Order::Greater;
            }
        }

        return Util::Order::Equal;
    }

    DEFINE_ARE_EQUAL(KnownConstant) {
        return BASE_ARE_EQUAL(KnownConstant) && symbol->known_constant.value == value;
    }

    DEFINE_COMPARE_TO(KnownConstant) {
        return Util::compare(
            static_cast<std::underlying_type_t<decltype(value)>>(value),
            static_cast<std::underlying_type_t<decltype(value)>>(other.as<KnownConstant>().value));
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
