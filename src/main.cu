#include <cstdlib>
#include <cstring>

#include <iostream>
#include <vector>

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "integral.cuh"
#include "integrate.cuh"
#include "symbol.cuh"

static constexpr size_t BLOCK_SIZE = 128;
static constexpr size_t BLOCK_COUNT = 32;

int main() {
    std::vector<Sym::Symbol> ixpr = Sym::integral(Sym::var() ^ Sym::num(2));
    std::cout << "ixpr1: " << ixpr[0].to_string() << std::endl;

    std::vector<Sym::Symbol> ixpr2 = Sym::substitute(ixpr, Sym::cos(Sym::var()));
    std::cout << "ixpr2: " << ixpr2[0].to_string() << std::endl;

    std::vector<Sym::Symbol> ixpr3 = Sym::substitute(ixpr2, Sym::e() ^ Sym::var());
    std::cout << "ixpr3: " << ixpr3[0].to_string() << std::endl;

    std::cout << "Creating an expression" << std::endl;
    std::vector<std::vector<Sym::Symbol>> expressions = {Sym::cos(Sym::var()),
                                                         Sym::sin(Sym::cos(Sym::var())),
                                                         Sym::e() ^ Sym::var(),
                                                         Sym::var() ^ Sym::num(5),
                                                         Sym::var() ^ Sym::pi(),
                                                         Sym::var() ^ Sym::var(),
                                                         Sym::pi()};

    for (size_t i = 0; i < expressions.size(); ++i) {
        std::cout << expressions[i][0].to_string() << std::endl;
    }

    std::cout << "Allocating and zeroing GPU memory" << std::endl;

    size_t mem_total = 0;

    Sym::Symbol* d_expressions;
    cudaMalloc(&d_expressions, Sym::EXPRESSION_ARRAY_SIZE * sizeof(Sym::Symbol));
    cudaMemset(d_expressions, 0, Sym::EXPRESSION_ARRAY_SIZE * sizeof(Sym::Symbol));
    mem_total += Sym::EXPRESSION_ARRAY_SIZE * sizeof(Sym::Symbol);

    Sym::Symbol* d_expressions_swap;
    cudaMalloc(&d_expressions_swap, Sym::EXPRESSION_ARRAY_SIZE * sizeof(Sym::Symbol));
    mem_total += Sym::EXPRESSION_ARRAY_SIZE * sizeof(Sym::Symbol);

    size_t* d_applicability;
    cudaMalloc(&d_applicability, Sym::APPLICABILITY_ARRAY_SIZE * sizeof(size_t));
    cudaMemset(d_applicability, 0, Sym::APPLICABILITY_ARRAY_SIZE * sizeof(size_t));
    mem_total += Sym::APPLICABILITY_ARRAY_SIZE * sizeof(size_t);

    size_t h_expression_count = expressions.size();
    size_t* d_expression_count;
    cudaMalloc(&d_expression_count, sizeof(size_t));
    mem_total += sizeof(size_t);

    std::cout << "Allocated " << mem_total << " bytes (" << mem_total / 1024 / 1024 << "MiB)" << std::endl;

    std::cout << "Copying to GPU memory" << std::endl;

    cudaMemcpy(d_expression_count, &h_expression_count, sizeof(size_t), cudaMemcpyHostToDevice);
    for (size_t i = 0; i < expressions.size(); ++i) {
        cudaMemcpy(d_expressions + Sym::EXPRESSION_MAX_SYMBOL_COUNT * i, expressions[i].data(),
                   expressions[i].size() * sizeof(Sym::Symbol), cudaMemcpyHostToDevice);
    }

    std::cout << "Checking heuristics" << std::endl;

    Sym::check_for_known_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(d_expressions, d_applicability,
                                                                d_expression_count);

    std::cout << "Calculating partial sum of applicability" << std::endl;

    thrust::inclusive_scan(thrust::device, d_applicability,
                           d_applicability + Sym::APPLICABILITY_ARRAY_SIZE, d_applicability);

    std::cout << "Applying heuristics" << std::endl;

    Sym::apply_known_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(d_expressions, d_expressions_swap,
                                                            d_applicability, d_expression_count);
    std::swap(d_expressions, d_expressions_swap);
    cudaMemcpy(d_expression_count, d_applicability + Sym::APPLICABILITY_ARRAY_SIZE - 1,
               sizeof(size_t), cudaMemcpyDeviceToDevice);

    std::cout << "Copying results to host memory" << std::endl;

    std::vector<size_t> h_applicability(Sym::APPLICABILITY_ARRAY_SIZE);
    cudaMemcpy(h_applicability.data(), d_applicability,
               Sym::APPLICABILITY_ARRAY_SIZE * sizeof(size_t), cudaMemcpyDeviceToHost);

    std::vector<Sym::Symbol> h_results(Sym::EXPRESSION_ARRAY_SIZE);
    cudaMemcpy(h_results.data(), d_expressions, Sym::EXPRESSION_ARRAY_SIZE * sizeof(Sym::Symbol),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_expression_count, d_expression_count, sizeof(size_t), cudaMemcpyDeviceToHost);

    std::cout << "Applicability:" << std::endl;
    for (size_t i = 0; i < h_applicability.size(); ++i) {
        if (i % Sym::MAX_EXPRESSION_COUNT == 0 && i != 0) {
            std::cout << std::endl;
        }

        std::cout << h_applicability[i] << ", ";
    }
    std::cout << std::endl;

    std::cout << "Results: " << std::endl;
    for (size_t i = 0; i < h_expression_count; ++i) {
        std::cout << h_results[i * Sym::EXPRESSION_MAX_SYMBOL_COUNT].to_string() << std::endl;
    }

    std::cout << "Freeing GPU memory" << std::endl;
    cudaFree(d_applicability);
    cudaFree(d_expressions_swap);
    cudaFree(d_expressions);
}
