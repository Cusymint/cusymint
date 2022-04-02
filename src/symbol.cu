#include "symbol.cuh"

namespace Sym {
    __host__ __device__ inline Type Symbol::type() const { return unknown.type; }
    __host__ __device__ inline bool Symbol::is(Type type) const { return unknown.type == type; }

    __host__ __device__ bool Symbol::is_simple_variable_polynomial() const {
        return this[0].is(Type::Power) && this[1].is(Type::Variable) &&
               this[2].is(Type::NumericConstant) && this[2].numeric_constant.value != 0.0;
    }

    __host__ __device__ bool Symbol::is_variable_reciprocal() const {
        return this[0].is(Type::Reciprocal) && this[1].is(Type::NumericConstant) &&
               this[2].is(Type::Variable);
    }

    __host__ __device__ bool Symbol::is_numeric_constant_addition() const {
        return this[0].is(Type::Addition) && this[1].is(Type::NumericConstant) &&
               this[2].is(Type::NumericConstant);
    }

    __host__ __device__ bool Symbol::is_numeric_constant_product() const {
        return this[0].is(Type::Product) && this[1].is(Type::NumericConstant) &&
               this[2].is(Type::NumericConstant);
    }

    __host__ __device__ bool Symbol::is_numeric_constant_negation() const {
        return this[0].is(Type::Negative) && this[1].is(Type::NumericConstant);
    }

    __host__ std::string Symbol::to_string() {
        switch (unknown.type) {
        case Type::Variable:
            return variable.to_string();
        case Type::NumericConstant:
            return numeric_constant.to_string();
        case Type::KnownConstant:
            return known_constant.to_string();
        case Type::UnknownConstant:
            return unknown_constant.to_string();
        case Type::Addition:
            return "(" + this[1].to_string() + "+" + this[addition.second_arg_offset].to_string() +
                   ")";
        case Type::Negative:
            return "-(" + this[1].to_string() + ")";
        case Type::Product:
            return "(" + this[1].to_string() + "*" + this[product.second_arg_offset].to_string() +
                   ")";
        case Type::Reciprocal:
            return "(1/(" + this[1].to_string() + "))";
        case Type::Power:
            return "(" + this[1].to_string() + "^" + this[power.second_arg_offset].to_string() +
                   ")";
        case Type::Sine:
            return "sin(" + this[1].to_string() + ")";
        case Type::Cosine:
            return "cos(" + this[1].to_string() + ")";
        case Type::Tangent:
            return "tan(" + this[1].to_string() + ")";
        case Type::Cotangent:
            return "cot(" + this[1].to_string() + ")";
        case Type::Unknown:
        default:
            return unknown.to_string();
        }
    }
}; // namespace Sym
