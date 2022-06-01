#include <cstdlib>
#include <cstring>

#include <iostream>
#include <vector>

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "integral.cuh"
#include "integrate.cuh"
#include "symbol.cuh"

static constexpr size_t BLOCK_SIZE = 1024;
static constexpr size_t BLOCK_COUNT = 32;

void test_substitutions() {
    std::cout << "Testing manual substitutions" << std::endl;
    std::vector<Sym::Symbol> ixpr = Sym::integral(Sym::var() ^ Sym::num(2));
    std::cout << "ixpr1: " << ixpr[0].to_string() << std::endl;

    std::vector<Sym::Symbol> ixpr2 = Sym::substitute(ixpr, Sym::cos(Sym::var()));
    std::cout << "ixpr2: " << ixpr2[0].to_string() << std::endl;

    std::vector<Sym::Symbol> ixpr3 = Sym::substitute(ixpr2, Sym::var() * (Sym::e() ^ Sym::var()));
    std::cout << "ixpr3: " << ixpr3[0].to_string() << std::endl;
}

void simplify(Sym::Symbol* const d_integrals, Sym::Symbol* const d_help_spaces,
              size_t* const d_integral_count) {
    std::cout << "Simplifying" << std::endl;

    Sym::simplify<<<BLOCK_COUNT, BLOCK_SIZE>>>(d_integrals, d_help_spaces, d_integral_count);
}

std::vector<std::vector<Sym::Symbol>> create_test_integrals() {
    std::cout << "Creating integrals" << std::endl;
    std::vector<std::vector<Sym::Symbol>> integrals{
        Sym::substitute(Sym::integral(Sym::cos(Sym::var())), Sym::pi() * Sym::var()),
        Sym::integral((Sym::var() ^ Sym::num(2.0)) ^ Sym::e()),
        Sym::integral(Sym::num(2.0) ^ Sym::num(3.0)),
        Sym::integral(Sym::cos(Sym::var()) ^ Sym::num(0.0)),
        Sym::integral(Sym::cos(Sym::sin(Sym::var()))),
        Sym::integral(Sym::e() ^ Sym::var()),
        Sym::integral((Sym::e() ^ Sym::var()) * (Sym::e() ^ Sym::var())),
        Sym::integral(Sym::var() ^ (-Sym::num(5.0))),
        Sym::integral(Sym::var() ^ (Sym::pi() - Sym::num(1.0))),
        Sym::integral(Sym::var() ^ Sym::var()),
        Sym::integral(Sym::pi() + Sym::e() * Sym::num(10.0)),
        Sym::integral((Sym::e() ^ Sym::var()) * (Sym::e() ^ (Sym::e() ^ Sym::var()))),
        Sym::integral(Sym::arccot(Sym::var())),
        Sym::integral(Sym::num(1.0) / ((Sym::var() ^ Sym::num(2.0)) + Sym::num(1.0))),
        Sym::integral(Sym::num(1.0) / (Sym::num(1.0) + (Sym::var() ^ Sym::num(2.0))))
        };

    for (size_t i = 0; i < integrals.size(); ++i) {
        std::cout << integrals[i][0].to_string() << std::endl;
    }
    std::cout << std::endl;

    return integrals;
}

void check_and_apply_heuristics(Sym::Symbol*& d_integrals, Sym::Symbol*& d_integrals_swap,
                                Sym::Symbol* const d_help_spaces, size_t* const d_integral_count,
                                size_t* const d_applicability) {
    std::cout << "Checking heuristics" << std::endl;

    cudaDeviceSynchronize();
    Sym::check_for_known_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(d_integrals, d_applicability,
                                                                d_integral_count);

    std::cout << "Calculating partial sum of applicability" << std::endl;

    cudaDeviceSynchronize();
    thrust::inclusive_scan(thrust::device, d_applicability,
                           d_applicability + Sym::APPLICABILITY_ARRAY_SIZE, d_applicability);

    std::cout << "Applying heuristics" << std::endl;

    cudaDeviceSynchronize();
    Sym::apply_known_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(
        d_integrals, d_integrals_swap, d_help_spaces, d_applicability, d_integral_count);

    std::swap(d_integrals, d_integrals_swap);
    cudaMemcpy(d_integral_count, d_applicability + Sym::APPLICABILITY_ARRAY_SIZE - 1,
               sizeof(size_t), cudaMemcpyDeviceToDevice);

    std::cout << std::endl;
}

void print_applicability(const size_t* const d_applicability, const size_t integral_count) {
    std::vector<size_t> h_applicability(Sym::APPLICABILITY_ARRAY_SIZE);
    cudaMemcpy(h_applicability.data(), d_applicability,
               Sym::APPLICABILITY_ARRAY_SIZE * sizeof(size_t), cudaMemcpyDeviceToHost);

    std::cout << "Applicability:" << std::endl;
    for (size_t i = 0; i < h_applicability.size(); ++i) {
        if (i % Sym::MAX_INTEGRAL_COUNT == 0 && i != 0) {
            std::cout << std::endl;
        }

        std::cout << h_applicability[i] << ", ";
    }
    std::cout << std::endl;
}

void print_results(const Sym::Symbol* const d_integrals, const size_t integral_count) {
    std::vector<Sym::Symbol> h_integrals(Sym::INTEGRAL_ARRAY_SIZE);
    cudaMemcpy(h_integrals.data(), d_integrals, Sym::INTEGRAL_ARRAY_SIZE * sizeof(Sym::Symbol),
               cudaMemcpyDeviceToHost);

    std::cout << "Results (" << integral_count << "):" << std::endl;
    for (size_t i = 0; i < integral_count; ++i) {
        std::cout << h_integrals[i * Sym::INTEGRAL_MAX_SYMBOL_COUNT].to_string() << std::endl;
    }
    std::cout << std::endl;
}

void print_current_results(const size_t* const d_applicability,
                           const Sym::Symbol* const d_integrals,
                           const size_t* const d_integral_count) {
    std::cout << "Copying results to host memory" << std::endl;

    size_t h_integral_count;
    cudaMemcpy(&h_integral_count, d_integral_count, sizeof(size_t), cudaMemcpyDeviceToHost);

    if (d_applicability != nullptr) {
        print_applicability(d_applicability, h_integral_count);
    }

    print_results(d_integrals, h_integral_count);
}

int main() {
    test_substitutions();

    std::vector<std::vector<Sym::Symbol>> integrals = create_test_integrals();

    std::cout << "Allocating and zeroing GPU memory" << std::endl;

    size_t mem_total = 0;

    Sym::Symbol* d_integrals;
    cudaMalloc(&d_integrals, Sym::INTEGRAL_ARRAY_SIZE * sizeof(Sym::Symbol));
    cudaMemset(d_integrals, 0, Sym::INTEGRAL_ARRAY_SIZE * sizeof(Sym::Symbol));
    mem_total += Sym::INTEGRAL_ARRAY_SIZE * sizeof(Sym::Symbol);

    Sym::Symbol* d_integrals_swap;
    cudaMalloc(&d_integrals_swap, Sym::INTEGRAL_ARRAY_SIZE * sizeof(Sym::Symbol));
    mem_total += Sym::INTEGRAL_ARRAY_SIZE * sizeof(Sym::Symbol);

    Sym::Symbol* d_help_spaces;
    cudaMalloc(&d_help_spaces, Sym::INTEGRAL_ARRAY_SIZE * sizeof(Sym::Symbol));
    mem_total += Sym::INTEGRAL_ARRAY_SIZE * sizeof(Sym::Symbol);

    size_t* d_applicability;
    cudaMalloc(&d_applicability, Sym::APPLICABILITY_ARRAY_SIZE * sizeof(size_t));
    cudaMemset(d_applicability, 0, Sym::APPLICABILITY_ARRAY_SIZE * sizeof(size_t));
    mem_total += Sym::APPLICABILITY_ARRAY_SIZE * sizeof(size_t);

    size_t h_integral_count = integrals.size();
    size_t* d_integral_count;
    cudaMalloc(&d_integral_count, sizeof(size_t));
    mem_total += sizeof(size_t);

    std::cout << "Allocated " << mem_total << " bytes (" << mem_total / 1024 / 1024 << "MiB)"
              << std::endl;

    std::cout << "Copying to GPU memory" << std::endl;

    cudaMemcpy(d_integral_count, &h_integral_count, sizeof(size_t), cudaMemcpyHostToDevice);
    for (size_t i = 0; i < integrals.size(); ++i) {
        cudaMemcpy(d_integrals + Sym::INTEGRAL_MAX_SYMBOL_COUNT * i, integrals[i].data(),
                   integrals[i].size() * sizeof(Sym::Symbol), cudaMemcpyHostToDevice);
    }

    std::cout << std::endl;

    simplify(d_integrals, d_help_spaces, d_integral_count);
    print_current_results(nullptr, d_integrals, d_integral_count);

    check_and_apply_heuristics(d_integrals, d_integrals_swap, d_help_spaces, d_integral_count,
                               d_applicability);
    print_current_results(d_applicability, d_integrals, d_integral_count);

    // cudaMemset(d_applicability, 0, Sym::APPLICABILITY_ARRAY_SIZE * sizeof(size_t));

    // check_and_apply_heuristics(d_integrals, d_integrals_swap, d_help_spaces, d_integral_count,
    //                            d_applicability);
    // print_current_results(d_applicability, d_integrals, d_integral_count);

    std::cout << "Freeing GPU memory" << std::endl;
    cudaFree(d_applicability);
    cudaFree(d_integrals_swap);
    cudaFree(d_integrals);
}
