#include "Polynomial.cuh"

#include <fmt/core.h>

#include "Addition.cuh"
#include "Macros.cuh"
#include "MetaOperators.cuh"
#include "Symbol.cuh"
#include "SymbolType.cuh"

#include "Utils/CompileConstants.cuh"

namespace {
    __host__ __device__ size_t size_from_rank(size_t rank) {
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
    DEFINE_NO_OP_PUSH_CHILDREN_ONTO_STACK(Polynomial)
    DEFINE_NO_OP_PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE(Polynomial)
    DEFINE_INVALID_IS_FUNCTION_OF(Polynomial)
    DEFINE_INVALID_DERIVATIVE(Polynomial)
    DEFINE_INVALID_SEAL_WHOLE(Polynomial)

    DEFINE_COMPARE_TO(Polynomial) {
        if (rank < other.as<Polynomial>().rank) {
            return Util::Order::Less;
        }

        if (rank > other.as<Polynomial>().rank) {
            return Util::Order::Greater;
        }

        for (size_t i = 0; i < rank; ++i) {
            const auto rank_compare = Util::compare(rank, other.as<Polynomial>().rank);

            if (rank_compare != Util::Order::Equal) {
                return rank_compare;
            }
        }

        return Util::Order::Equal;
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

    DEFINE_ARE_EQUAL(Polynomial) {
        return BASE_ARE_EQUAL(Polynomial) && symbol->as<Polynomial>().rank == rank &&
               are_coefficients_equal(*this, symbol->as<Polynomial>());
    }

    __host__ __device__ void Polynomial::make_polynomial_at(const Symbol* const symbol,
                                                            Symbol* const destination) {
        auto* const term_ranks = reinterpret_cast<Util::OptionalNumber<ssize_t>*>(destination);
        auto* const term_coefficients_dst =
            destination + symbol->size() * sizeof(Util::OptionalNumber<ssize_t>) / sizeof(Symbol) +
            1;
        auto* const term_coefficients =
            reinterpret_cast<Util::OptionalNumber<double>*>(term_coefficients_dst);
        auto* const dst_coefs = reinterpret_cast<double*>(term_coefficients + symbol->size());

        symbol->is_polynomial(destination);
        symbol->get_monomial_coefficient(term_coefficients_dst);

        const size_t rank = term_ranks[0].value();

        for (ssize_t i = 0; i <= rank; ++i) {
            dst_coefs[i] = 0;
        }

        switch (symbol->type()) {
        case Type::Addition: {
            ConstTreeIterator<Addition> iterator(symbol->as_ptr<Addition>());
            while (iterator.is_valid()) {
                const size_t offset = iterator.current() - symbol;
                dst_coefs[term_ranks[offset].value()] = term_coefficients[offset].value();

                iterator.advance();
            }
        } break;
        case Type::Product:
            [[fallthrough]];
        case Type::Power:
            [[fallthrough]];
        case Type::Negation:
            [[fallthrough]];
        case Type::Variable:
            dst_coefs[rank] = term_coefficients[0].value();
            break;
        default:
            Util::crash("Improper use of make_polynomial_at() function on symbol type %s.",
                        type_name(symbol->type()));
        }

        auto& dest_poly = destination->init_from(Polynomial::with_rank(rank));

        double* const coefs = dest_poly.coefficients();

        for (ssize_t i = 0; i <= rank; ++i) {
            coefs[i] = dst_coefs[i];
        }
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

        for (ssize_t i = static_cast<ssize_t>(rank) - 1; i >= 0; --i) {
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
        for (auto i = static_cast<ssize_t>(numerator.rank - denominator.rank); i >= 0; --i) {
            double& num_first = numerator[i + denominator.rank];
            double& res_current = result[i];
            res_current = num_first / denominator[denominator.rank];
            num_first = 0;
            for (ssize_t j = static_cast<ssize_t>(denominator.rank) - 1; j >= 0; --j) {
                numerator[i + j] -= res_current * denominator[j];
            }
        }
        numerator.make_proper();
    }

    __host__ __device__ void Polynomial::make_proper() {
        auto i = static_cast<ssize_t>(rank);
        while (i > 0 && abs(coefficients()[i--]) < Consts::EPS) {
            --rank;
        }
        size = size_from_rank(rank);
    }

    std::string Polynomial::to_string() const {
        std::string coefficients_str =
            fmt::format(rank == 0 ? "Poly[size={}]({}"
                                  : (rank == 1 ? "Poly[size={}]({}*x" : "Poly[size={}]({}*x^{}"),
                        size, coefficients()[rank], rank);
        for (auto i = static_cast<ssize_t>(rank) - 1; i > 1; --i) {
            if (coefficients()[i] != 0) {
                coefficients_str += fmt::format("{}{}*x^{}", coefficients()[i] < 0 ? "" : "+",
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

    std::string Polynomial::to_tex() const {
        std::string coefficients_str = fmt::format(
            rank == 0 ? "{}" : (rank == 1 ? "{}x" : "{}x^{{ {} }}"), coefficients()[rank], rank);
        for (auto i = static_cast<ssize_t>(rank) - 1; i > 1; --i) {
            if (coefficients()[i] != 0) {
                coefficients_str += fmt::format("{}{}x^{{ {} }}", coefficients()[i] < 0 ? "" : "+",
                                                coefficients()[i], i);
            }
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
