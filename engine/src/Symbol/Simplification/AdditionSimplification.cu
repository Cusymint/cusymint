#include "AdditionSimplification.cuh"
#include "Symbol/Constants.cuh"
#include "Symbol/Product.cuh"
#include "Symbol/SimplificationResult.cuh"
#include "Symbol/Symbol.cuh"

namespace {
    __host__ __device__ void add_symbols(Sym::Symbol& augend, Sym::Symbol& addend, bool is_mul_addend) {
        double& coefficient = augend.as<Sym::Product>().arg1().as<Sym::NumericConstant>().value;
        const double value = is_mul_addend ? addend.as<Sym::Product>().arg1().as<Sym::NumericConstant>().value : 1;
        if ((coefficient += value) == 0) {
            augend.init_from(Sym::NumericConstant::with_value(0));
        }
        addend.init_from(Sym::NumericConstant::with_value(0));
    }
}

namespace Sym {
    __host__ __device__ SimplificationResult try_add_symbols(Symbol& expr1, Symbol& expr2) {
        const bool is_mul1 = Mul<Num, Any>::match(expr1);
        const bool is_mul2 = Mul<Num, Any>::match(expr2);
        Symbol& base1 = is_mul1 ? expr1.as<Product>().arg2() : expr1;
        Symbol& base2 = is_mul2 ? expr2.as<Product>().arg2() : expr2;

        if (!Symbol::are_expressions_equal(base1, base2)) {
            return SimplificationResult::NoAction;
        }

        if (is_mul1) {
            add_symbols(expr1, expr2, is_mul2);
            return SimplificationResult::Success;
        }

        if (is_mul2) {
            add_symbols(expr2, expr1, is_mul1);
            return SimplificationResult::Success;
        }

        // TODO
    }
}