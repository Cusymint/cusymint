#ifndef BRING_OUT_CONSTANTS_FROM_PRODUCT_CUH
#define BRING_OUT_CONSTANTS_FROM_PRODUCT_CUH

#include "Heuristic.cuh"

namespace Sym::Heuristic {
    __device__ CheckResult contains_constants_product(const Integral& integral);
    __device__ void bring_out_constants_from_product(
        const SubexpressionCandidate& integral, const ExpressionArray<>::Iterator& integral_dst,
        const ExpressionArray<>::Iterator& expression_dst, Symbol& /*help_space*/);
}

#endif