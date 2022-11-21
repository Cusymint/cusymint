#ifndef UNIVERSAL_SUBSTITUTION_CUH
#define UNIVERSAL_SUBSTITUTION_CUH

#include "Heuristic.cuh"

namespace Sym::Heuristic {
    __device__ CheckResult is_function_of_trigs(const Integral& integral);
    __device__ void do_universal_substitution(const SubexpressionCandidate& integral,
                                              const ExpressionArray<>::Iterator& integral_dst,
                                              const ExpressionArray<>::Iterator& /*expression_dst*/,
                                              Symbol& help_space);
}
#endif