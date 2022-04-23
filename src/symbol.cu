#include "symbol.cuh"

namespace Sym {
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

    void Symbol::substitute_variable_with(Symbol symbol) {
        for (size_t i = 0; i < unknown.total_size; ++i) {
            if (this[i].is(Type::Variable)) {
                this[i] = symbol;
            }
        }
    }

    std::string Symbol::to_string() { return VIRTUAL_CALL(*this, to_string); }

    __host__ __device__ void simplify_in_place() {}
};
