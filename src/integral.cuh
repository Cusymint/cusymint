#ifndef INTEGRAL_CUH
#define INTEGRAL_CUH

#include <vector>

#include "substitution.cuh"
#include "symbol_defs.cuh"

namespace Sym {
    DECLARE_SYMBOL(Integral, false)
    size_t substitution_count;
    size_t integrand_offset;
    std::string to_string() const;

    __host__ __device__ Symbol* integrand();
    __host__ __device__ const Symbol* integrand() const;

    /*
     * @brief Copies `*this`, and all its substitutions into dst, and adds expression from
     * `substitution` as a new substitution at the end. Updates substitution_cont, integrand_offset
     * all `sub_substitution_count`s. Does not update `total_size`s, neither in the integral, nor in
     * substitutions. Assigns correct substitution_idx in the last substitution.
     *
     * @param substitution Expression to be used as substitution (without `Substitution` symbol)
     * @param destination Destination to copy everything to
     */
    __host__ __device__ void
    copy_substitutions_with_an_additional_one(const Symbol* const substitution,
                                              Symbol* const destination) const;

    /*
     * @brief Integrate `this` by substitution and save the result in `destination`.
     *
     *
     * @param substitution Expression that will be replaced by variable in `this`, e.g. if u=x^2
     * then `substitution`=var^2
     * @param derivative Derivative of `substitution` in terms of u, e.g. if u=x^2 then
     * `derivative`=2*(var^(1/2))
     * @param destination Pointer to where the result is going to be saved
     */
    __host__ __device__ void integrate_by_substitution_with_derivative(
        const Symbol* const substitution, const Symbol* const derivative, Symbol* const destination,
        Symbol* const swap_space) const;

    __host__ __device__ Substitution* first_substitution();
    END_DECLARE_SYMBOL(Integral)

    std::vector<Symbol> integral(const std::vector<Symbol>& arg);
}

#endif
