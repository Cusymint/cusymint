#ifndef INTEGRATE_CUH
#define INTEGRATE_CUH

#include <optional>

#include "Symbol/ExpressionArray.cuh"

#include "Symbol/Symbol.cuh"

#include <optional>

namespace Sym {
    /*
     * @brief Solves an integral and returns the result
     *
     * @param integral Vector of symbols with the integral, the first symbol should be
     * `Sym::Integral`
     *
     * @return `std::nullopt` if no result has been found, vector of vectors with the solution tree
     * otherwise
     */
    std::optional<std::vector<std::vector<Sym::Symbol>>>
    solve_integral(const std::vector<Sym::Symbol>& integral);
}

#endif
