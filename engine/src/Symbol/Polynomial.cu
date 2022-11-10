#include "Macros.cuh"
#include "MetaOperators.cuh"
#include "Polynomial.cuh"
#include "Symbol.cuh"
#include "Symbol/Addition.cuh"
#include "Utils/Cuda.cuh"
#include <fmt/core.h>

namespace {
    __host__ __device__ inline size_t size_from_rank(size_t rank) {
        return 2 + sizeof(double) * (rank + 1) / sizeof(Sym::Symbol);
    }
}

namespace Sym {
    DEFINE_COMPRESS_REVERSE_TO(Polynomial) {
        for (size_t i = 0; i < additional_required_size; ++i) {
            (destination + i)->init_from(Unknown::create());
        }
        Symbol* const new_destination = destination + additional_required_size;
        Symbol::copy_and_reverse_symbol_sequence(new_destination, symbol(), size);
        return size + additional_required_size;
    }

    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(Polynomial)
    DEFINE_INTO_DESTINATION_OPERATOR(Polynomial)
    DEFINE_NO_OP_PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE(Polynomial)

    DEFINE_IS_FUNCTION_OF(Polynomial) {
        for (size_t i = 0; i < expression_count; ++i) {
            if (!expressions[i]->is(Type::Variable) && !expressions[i]->is_constant()) {
                return false;
            }
        }
        return true;
    }

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

    __host__ __device__ void Polynomial::expand_to(Symbol* destination) const {

        Symbol* coefficient_dst = destination + rank;

        Num::init(*coefficient_dst, {coefficients()[0]});
        coefficient_dst += coefficient_dst->size();

        if (rank >= 1) {
            Mul<Num, Var>::init(*coefficient_dst, {coefficients()[1]});
            coefficient_dst += coefficient_dst->size();
        }

        for (size_t i = 2; i <= rank; ++i) {
            Mul<Num, Pow<Var, Num>>::init(*coefficient_dst, {coefficients()[i], i});
            coefficient_dst += coefficient_dst->size();
        }

        for (ssize_t i = rank - 1; i >= 0; --i) {
            Addition* const plus = (destination + i) << Addition::builder();
            plus->seal_arg1();
            plus->seal();
        }
    }

    __host__ __device__ inline size_t Polynomial::expanded_size() const {
        return expanded_size_from_rank(rank);
    }

    __host__ __device__ size_t Polynomial::expanded_size_from_rank(size_t rank) {
        return rank == 0 ? 1 : (6 * rank - 1); 
    }

    __host__ __device__ Polynomial Polynomial::with_rank(size_t rank) {
        return {
            .type = Type::Polynomial,
            .size = size_from_rank(rank),
            .simplified = true,
            .rank = rank,
        };
    }

    __host__ __device__ void Polynomial::divide_polynomials(Polynomial& numerator,
                                                            Polynomial& denominator,
                                                            Polynomial& result) {        
        // for (ssize_t i=0;i<=result.rank;++i) {
        //     printf("%f\t",result[i]);
        // }
        // printf("\n");
        
        for (ssize_t i = numerator.rank - denominator.rank; i >= 0; --i) {
            double& num_first = numerator[i + denominator.rank];
            double& res_current = result[i];
            res_current = num_first / denominator[denominator.rank];
            num_first = 0;
            for (ssize_t j = denominator.rank - 1; j >= 0; --j) {
                numerator[i + j] -= res_current * denominator[j];
            }
        //     for (ssize_t i=0;i<=result.rank;++i) {
        //     printf("%f\t",result[i]);
        // }
        // printf("\n");
        }
        numerator.make_proper();
    }

    __host__ __device__ void Polynomial::make_proper() {
        ssize_t i = rank;
        while (i > 0 && abs(coefficients()[i--]) < Util::EPS) {
            --rank;
        }
        size = size_from_rank(rank);
    }

    std::string Polynomial::to_string() const { // TODO lepiej!
        std::string coefficients_str =
            fmt::format(rank == 0 ? "Poly[size={}]({}"
                                  : (rank == 1 ? "Poly[size={}]({}^x" : "Poly[size={}]({}x^({})"),
                        size, coefficients()[rank], rank);
        for (int i = rank - 1; i > 1; --i) {
            if (coefficients()[i] != 0) {
                coefficients_str += fmt::format("{}{}*x^({})", coefficients()[i] < 0 ? "" : "+",
                                                coefficients()[i], i);
            }
        }
        if (rank > 1 && coefficients()[1] != 0) {
            coefficients_str +=
                fmt::format("{}{}*x", coefficients()[1] < 0 ? "" : "+", coefficients()[1]);
        }
        if (rank > 0 && coefficients()[0] != 0) {
            coefficients_str +=
                fmt::format("{}{}", coefficients()[0] < 0 ? "" : "+", coefficients()[0]);
        }
        return coefficients_str + ")";
    }

    std::string Polynomial::to_tex() const { // TODO popraw!
        std::string coefficients_str = fmt::format("{}x^{{ {} }}", coefficients()[rank], rank);
        for (int i = rank - 1; i >= 0; --i) {
            coefficients_str += fmt::format("{}{}x^{{ {} }}", coefficients()[i] < 0 ? "" : "+",
                                            coefficients()[i], i);
        }
        if (rank > 0 && coefficients()[1] != 0) {
            coefficients_str +=
                fmt::format("{}{}x", coefficients()[1] < 0 ? "" : "+", coefficients()[1]);
        }
        if (coefficients()[0] != 0) {
            coefficients_str +=
                fmt::format("{}{}", coefficients()[0] < 0 ? "" : "+", coefficients()[0]);
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