#ifndef STATIC_FUNCTIONS_CUH
#define STATIC_FUNCTIONS_CUH

#include "Utils/Cuda.cuh"

#include "Symbol/Symbol.cuh"

namespace Sym::Static {
    __device__ const Symbol& identity();
    __device__ const Symbol& inverse();

    __device__ const Symbol& sin_x();
    __device__ const Symbol& cos_x();
    __device__ const Symbol& tan_x();
    __device__ const Symbol& cot_x();

    // Functions used in the universal substitution
    __device__ const Symbol& universal_sin_x();
    __device__ const Symbol& universal_cos_x();
    __device__ const Symbol& universal_tan_x();
    __device__ const Symbol& universal_cot_x();
    __device__ const Symbol& universal_derivative();
    __device__ const Symbol& tan_x_over_2();

    __device__ const Symbol& e_to_x();

    // sqrt(1-x^2)
    __device__ const Symbol& pythagorean_sin_cos();
    // -sqrt(1-x^2)
    __device__ const Symbol& neg_pythagorean_sin_cos();
    // x/sqrt(1-x^2)
    __device__ const Symbol& tangent_as_sine();
    // sqrt(1-x^2)/x
    __device__ const Symbol& cotangent_as_sine();

    /*
     * @brief Initializes static functions used by Cusymint. Has to be called before any integration
     * takes place.
     */
    void init_functions();
}

#endif
