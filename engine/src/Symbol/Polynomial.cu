#include "Macros.cuh"
#include "Polynomial.cuh"
#include "Symbol.cuh"
#include "Utils/Cuda.cuh"
#include <fmt/core.h>

namespace Sym {
    DEFINE_COMPRESS_REVERSE_TO(Polynomial) {
        Symbol::copy_and_reverse_symbol_sequence(destination, this_symbol(), size);
        return size;
    }

    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(Polynomial)
    DEFINE_INTO_DESTINATION_OPERATOR(Polynomial)

    __host__ __device__ bool are_coefficients_equal(const Polynomial& poly1,
                                                    const Polynomial& poly2) {
        // Assumption: polynomials have the same rank
        for (int i = 0; i <= poly1.rank; ++i) {
            if (poly1[i] != poly2[i]) {
                return false;
            }
        }
        return true;
    }

    DEFINE_COMPARE(Polynomial) {
        return BASE_COMPARE(Polynomial) && symbol->polynomial.rank == rank &&
               are_coefficients_equal(*this, symbol->polynomial);
    }

    // __host__ __device__ void Polynomial::seal() {
    //     size = 2 + sizeof(double) * (rank + 1) / sizeof(Symbol); // non-optimal (there is a
    //     mistake - when rank is big!)
    // }

    __host__ __device__ Polynomial Polynomial::with_rank(int rank) {
        return {
            .type = Type::Polynomial,
            .size = 2 + sizeof(double) * (rank + 1) / sizeof(Symbol),
            .simplified = true,
            .rank = rank,
        };
    }

    std::string Polynomial::to_string() const {
        std::string coefficients_str =
            fmt::format("Poly[size={}]({}*x^({})", size, coefficients()[rank], rank);
        for (int i = rank - 1; i > 1; --i) {
            if (coefficients()[i] != 0) {
                coefficients_str += fmt::format("{}{}*x^({})", coefficients()[i] < 0 ? "" : "+",
                                                coefficients()[i], i);
            }
        }
        if (rank > 0 && coefficients()[1] != 0) {
            coefficients_str += fmt::format("{}{}*x", coefficients()[1] < 0 ? "" : "+",
                                                coefficients()[1]);
        }
        if (coefficients()[0] != 0) {
            coefficients_str += fmt::format("{}{}", coefficients()[0] < 0 ? "" : "+",
                                                coefficients()[0]);
        }
        return coefficients_str + ")";
    }

    std::string Polynomial::to_tex() const {
        std::string coefficients_str = fmt::format("{}x^{{ {} }}", coefficients()[rank], rank);
        for (int i = rank - 1; i >= 0; --i) {
            coefficients_str += fmt::format("{}{}x^{{ {} }}", coefficients()[i] < 0 ? "" : "+",
                                            coefficients()[i], i);
        }
        if (rank > 0 && coefficients()[1] != 0) {
            coefficients_str += fmt::format("{}{}x", coefficients()[1] < 0 ? "" : "+",
                                                coefficients()[1]);
        }
        if (coefficients()[0] != 0) {
            coefficients_str += fmt::format("{}{}", coefficients()[0] < 0 ? "" : "+",
                                                coefficients()[0]);
        }
        return coefficients_str;
    }

    __host__ __device__ double* Polynomial::coefficients() {
        return reinterpret_cast<double*>(&(Symbol::from(this)[1]));
    }

    __host__ __device__ const double* Polynomial::coefficients() const {
        return reinterpret_cast<const double*>(&(Symbol::from(this)[1]));
    }
}