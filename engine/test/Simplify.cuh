#ifndef TEST_SIMPLIFY_CUH
#define TEST_SIMPLIFY_CUH

#include <vector>

#include "Evaluation/Integrator.cuh"
#include "Symbol/Symbol.cuh"

namespace Test {
    void simplify_vector(std::vector<Sym::Symbol>& expression);
}

#endif
