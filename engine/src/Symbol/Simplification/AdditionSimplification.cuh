#include "Symbol/SimplificationResult.cuh"
#include "Symbol/Symbol.cuh"
#include "Symbol/MetaOperators.cuh"

namespace Sym {
    __host__ __device__ SimplificationResult try_add_symbols(Symbol& expr1, Symbol& expr2);
}