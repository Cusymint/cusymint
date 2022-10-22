#ifndef INTEGRAL_CUH
#define INTEGRAL_CUH

#include <vector>

#include "Macros.cuh"
#include "Substitution.cuh"

namespace Sym {
    DECLARE_SYMBOL(Integral, false)
    size_t substitution_count;
    size_t integrand_offset;

    __host__ __device__ void seal_no_substitutions();
    __host__ __device__ void seal_single_substitution();
    __host__ __device__ void seal_substitutions(const size_t count, const size_t size);

    [[nodiscard]] __host__ __device__ Symbol* integrand();
    [[nodiscard]] __host__ __device__ const Symbol* integrand() const;

    /*
     * @brief Copies `*this`, and all its substitutions into dst, and adds expression from
     * `substitution` as a new substitution at the end. Updates substitution_cont, integrand_offset.
     * Does not update `size`s, neither in the integral, nor in substitutions.
     * Assigns correct substitution_idx in the last substitution.
     *
     * @param substitution Expression to be used as substitution (without `Substitution` symbol)
     * @param destination Destination to copy everything to
     */
    __host__ __device__ void
    copy_substitutions_with_an_additional_one(const Symbol* const substitution_expr,
                                              Symbol* const destination) const;

    /*
     * @brief Copies the integral symbol and its substitutions to `destination`
     *
     * @param destination Destination of
     */
    __host__ __device__ void copy_without_integrand_to(Symbol* const destination) const;

    /*
     * @brief Integrate `this` by substitution and save the result in `destination`.
     *
     *
     * @param substitution Expression that will be replaced by variable in `this`, e.g. if u=x^2
     * then `substitution`=var^2
     * @param derivative Derivative of `substitution` in terms of u, e.g. if u=x^2 then
     * `derivative`=2*(var^(1/2))
     * @param destination Pointer to where the result is going to be saved
     * @param help_space Additional memory for internal usage
     */
    __host__ __device__ void integrate_by_substitution_with_derivative(
        const Symbol* const substitution, const Symbol* const derivative, Symbol* const destination,
        Symbol* const help_space) const;

    [[nodiscard]] __host__ __device__ const Substitution* first_substitution() const;
    [[nodiscard]] __host__ __device__ Substitution* first_substitution();

    /*
     * @brief Number of symbols between the integral symbol and the integrand (not counting either
     * of them)
     */
    [[nodiscard]] __host__ __device__ size_t substitutions_size() const;

    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::string to_tex() const;
    END_DECLARE_SYMBOL(Integral)

    std::vector<Symbol> integral(const std::vector<Symbol>& arg);
}

#endif
