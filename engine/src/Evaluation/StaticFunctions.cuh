#ifndef STATIC_FUNCTIONS_CUH
#define STATIC_FUNCTIONS_CUH

#include "Utils/Cuda.cuh"

#include "Symbol/Symbol.cuh"

namespace Sym::Static {
    __device__ const Symbol& identity();
    __device__ const Symbol& sin_x();
    __device__ const Symbol& cos_x();
    __device__ const Symbol& e_to_x();

    /*
     * @brief Initializes static functions used by Cusymint. Has to be called before any integration
     * takes place.
     */
    void init_functions();
}

#endif
