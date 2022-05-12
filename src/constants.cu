#include "constants.cuh"

#include <stdexcept>

#include "cuda_utils.cuh"
#include "symbol.cuh"

namespace Sym {
    DEFINE_SIMPLE_COMPRESS_REVERSE_TO(NumericConstant);
    DEFINE_SIMPLE_COMPRESS_REVERSE_TO(KnownConstant);
    DEFINE_SIMPLE_COMPRESS_REVERSE_TO(UnknownConstant);

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

    UnknownConstant UnknownConstant::create(const char* const name) {
        UnknownConstant unknown_constant = UnknownConstant::create();
        if (strlen(name) + 1 > NAME_LEN) {
            throw std::invalid_argument("UnknownConstant created with name longer than NAME_LEN");
        }

        strcpy(unknown_constant.name, name);
        return unknown_constant;
    }

    std::vector<Symbol> known_constant(KnownConstantValue value) {
        std::vector<Symbol> v(1);
        v[0].known_constant = KnownConstant::create();
        v[0].known_constant.value = value;
        return v;
    }

    std::vector<Symbol> e() { return known_constant(KnownConstantValue::E); }

    std::vector<Symbol> pi() { return known_constant(KnownConstantValue::Pi); }

    std::vector<Symbol> cnst(const char name[UnknownConstant::NAME_LEN]) {
        std::vector<Symbol> v(1);
        v[0].unknown_constant = UnknownConstant::create();
        std::copy(name, name + UnknownConstant::NAME_LEN, v[0].unknown_constant.name);
        return v;
    }

    std::vector<Symbol> num(double value) {
        std::vector<Symbol> v(1);
        v[0].numeric_constant = NumericConstant::create();
        v[0].numeric_constant.value = value;
        return v;
    }
}
