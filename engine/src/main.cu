#include <cstdlib>
#include <cstring>

#include <fmt/core.h>
#include <iostream>
#include <vector>

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "Evaluation/Integrate.cuh"

#include "Symbol/Constants.cuh"
#include "Symbol/ExpressionArray.cuh"
#include "Symbol/Integral.cuh"
#include "Symbol/Symbol.cuh"
#include "Symbol/Variable.cuh"

static constexpr size_t BLOCK_SIZE = 1024;
static constexpr size_t BLOCK_COUNT = 32;

void test_substitutions() {
    fmt::print("Testing manual substitutions\n");

    std::vector<Sym::Symbol> ixpr = Sym::integral(Sym::var() ^ Sym::num(2));
    fmt::print("Expression 1: {}\n", ixpr.data()->to_string());

    std::vector<Sym::Symbol> ixpr2 = Sym::substitute(ixpr, Sym::cos(Sym::var()));
    fmt::print("Expression 2: {}\n", ixpr2.data()->to_string());

    std::vector<Sym::Symbol> ixpr3 = Sym::substitute(ixpr2, Sym::var() * (Sym::e() ^ Sym::var()));
    fmt::print("Expression 3: {}\n", ixpr3.data()->to_string());
}

void simplify_integrals(Sym::ExpressionArray<Sym::Integral>& integrals,
                        Sym::ExpressionArray<>& help_spaces) {
    fmt::print("Simplifying\n");

    Sym::simplify<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, help_spaces);
}

std::vector<std::vector<Sym::Symbol>> create_test_integrals() {
    fmt::print("Creating integrals\n");
    std::vector<std::vector<Sym::Symbol>> integrals{
        // Sym::substitute(Sym::integral(Sym::cos(Sym::var())), Sym::pi() * Sym::var()),
        // Sym::integral((Sym::sin(Sym::var()) ^ Sym::num(2.0)) +
        //               (Sym::num(-8) + (Sym::cos(Sym::var()) ^ Sym::num(2.0)) + Sym::num(4))),
        // Sym::integral((Sym::var() + Sym::num(1.0)) +
        //               (Sym::pi() + (Sym::e() + Sym::cos(Sym::var())))),
        // Sym::integral((Sym::var() ^ Sym::num(2.0)) ^ Sym::e()),
        // Sym::integral(Sym::num(2.0) ^ Sym::num(3.0)),
        // Sym::integral(Sym::cos(Sym::var()) ^ Sym::num(0.0)),
        // Sym::integral(Sym::cos(Sym::sin(Sym::var()))),
        // Sym::integral(Sym::e() ^ Sym::var()),
        // Sym::integral((Sym::e() ^ Sym::var()) * (Sym::e() ^ Sym::var())),
        Sym::integral(Sym::var() ^ (-Sym::num(5.0))),
        Sym::integral(Sym::var() ^ (Sym::pi() - Sym::num(1.0))),
        Sym::integral(Sym::var() ^ Sym::var()),
        // Sym::integral(Sym::pi() + Sym::e() * Sym::num(10.0)),
        // Sym::integral((Sym::e() ^ Sym::var()) * (Sym::e() ^ (Sym::e() ^ Sym::var()))),
        // Sym::integral(Sym::arccot(Sym::var())),
        // Sym::integral(Sym::num(1.0) / ((Sym::var() ^ Sym::num(2.0)) + Sym::num(1.0))),
        // Sym::integral(Sym::num(1.0) / (Sym::num(1.0) + (Sym::var() ^ Sym::num(2.0)))),
        // Sym::integral(Sym::sinh(Sym::var())),
        Sym::integral(-Sym::num(-5.0)),
        Sym::integral(((Sym::var()^Sym::num(10))-Sym::num(3))/(Sym::var()+Sym::num(1))),
        // Sym::integral((Sym::var()^Sym::num(2))+(Sym::var()^Sym::num(3))+Sym::num(5)*(Sym::var()^Sym::num(2))),
        // Sym::integral((Sym::var()^Sym::num(2))+(Sym::var()^Sym::num(2))+Sym::num(5)*(Sym::var()^Sym::num(3))),
        // Sym::integral(Sym::num(8)*(Sym::var()^Sym::num(2))+(Sym::var()^Sym::num(2))+Sym::num(5)*(Sym::var()^Sym::num(3)))
        };

    for (const auto& integral : integrals) {
        fmt::print("{}\n", integral.data()->to_string());
    }
    fmt::print("\n");

    return integrals;
}

void check_and_apply_heuristics(Sym::ExpressionArray<Sym::Integral>& integrals,
                                Sym::ExpressionArray<Sym::Integral>& integrals_swap,
                                Sym::ExpressionArray<>& help_spaces,
                                Util::DeviceArray<size_t>& applicability) {
    fmt::print("Creating heuristics\n");

    cudaDeviceSynchronize();
    Sym::check_heuristics_applicability<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, applicability);

    cudaDeviceSynchronize();
    thrust::inclusive_scan(thrust::device, applicability.begin(), applicability.end(),
                           applicability.data());

    cudaDeviceSynchronize();
    Sym::apply_heuristics<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, integrals_swap, help_spaces,
                                                       applicability);
    integrals.resize_from_device(applicability.last());
    integrals_swap.resize(integrals.size());
    help_spaces.resize(integrals.size());

    std::swap(integrals, integrals_swap);

    fmt::print("\n");
}

void print_applicability(const Util::DeviceArray<size_t>& applicability) {
    const auto h_applicability = applicability.to_vector();

    fmt::print("Applicability:\n");
    for (size_t i = 0; i < h_applicability.size(); ++i) {
        if (i % Sym::MAX_INTEGRAL_COUNT == 0 && i != 0) {
            fmt::print("\n");
        }

        fmt::print("{}, ", h_applicability[i]);
    }
    fmt::print("\n");
}

void check_and_apply_known_itegrals(Sym::ExpressionArray<Sym::Integral>& integrals,
                                    Sym::ExpressionArray<Sym::Integral>& integrals_swap,
                                    Sym::ExpressionArray<>& help_spaces,
                                    Util::DeviceArray<size_t>& applicability) {
    fmt::print("Checking for known integrals\n");

    cudaDeviceSynchronize();
    Sym::check_for_known_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, applicability);

    cudaDeviceSynchronize();
    thrust::inclusive_scan(thrust::device, applicability.begin(), applicability.end(),
                           applicability.data());

    cudaDeviceSynchronize();
    Sym::apply_known_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, integrals_swap, help_spaces,
                                                            applicability);
    integrals.resize_from_device(applicability.last());
    integrals_swap.resize(integrals.size());
    help_spaces.resize(integrals.size());

    std::swap(integrals, integrals_swap);

    fmt::print("\n");
}

void print_results(const Sym::ExpressionArray<Sym::Integral> integrals) {
    const auto h_integrals = integrals.to_vector();

    fmt::print("Results:({}):\n", integrals.size());
    for (size_t int_idx = 0; int_idx < integrals.size(); ++int_idx) {
        fmt::print("{}\n", h_integrals[int_idx].data()->to_string());
    }

    fmt::print("\n");
}

void print_results_tex(const Sym::ExpressionArray<Sym::Integral> integrals) {
    const auto h_integrals = integrals.to_vector();

    fmt::print("Results:({}):\n", integrals.size());
    for (size_t int_idx = 0; int_idx < integrals.size(); ++int_idx) {
        fmt::print("{} \\\\ \n", h_integrals[int_idx].data()->to_tex());
    }

    fmt::print("\n");
}

void print_polynomial_ranks(const Sym::ExpressionArray<Sym::Integral> integrals) {
    const auto h_integrals = integrals.to_vector();

    fmt::print("Polynomial ranks:({}):\n", integrals.size());
    for (size_t int_idx = 0; int_idx < integrals.size(); ++int_idx) {
        fmt::print("{}: {}\n", int_idx, h_integrals[int_idx].data()->as<Sym::Integral>().integrand()->is_polynomial());
    }

    fmt::print("\n");
}

int main() {
    std::vector<std::vector<Sym::Symbol>> h_integrals = create_test_integrals();

    fmt::print("Allocating and zeroing GPU memory\n\n");

    Sym::ExpressionArray<Sym::Integral> integrals(h_integrals, Sym::INTEGRAL_MAX_SYMBOL_COUNT,
                                                  Sym::MAX_INTEGRAL_COUNT);
    Sym::ExpressionArray<Sym::Integral> integrals_swap(Sym::INTEGRAL_MAX_SYMBOL_COUNT,
                                                       Sym::MAX_INTEGRAL_COUNT, h_integrals.size());
    Sym::ExpressionArray<> help_spaces(Sym::INTEGRAL_MAX_SYMBOL_COUNT, Sym::MAX_INTEGRAL_COUNT,
                                       h_integrals.size());

    Util::DeviceArray<size_t> applicability(Sym::APPLICABILITY_ARRAY_SIZE, true);

    print_results_tex(integrals);

    simplify_integrals(integrals, help_spaces);
    print_results(integrals);

    print_polynomial_ranks(integrals);

    check_and_apply_heuristics(integrals, integrals_swap, help_spaces, applicability);
    print_results(integrals);
    applicability.zero_mem();

    simplify_integrals(integrals, help_spaces);
    print_results(integrals);

    check_and_apply_known_itegrals(integrals, integrals_swap, help_spaces, applicability);
    print_results(integrals);
    applicability.zero_mem();

    simplify_integrals(integrals, help_spaces);
    print_results(integrals);
}
