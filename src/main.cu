#include <cstdlib>
#include <cstring>

#include <iostream>
#include <vector>

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "integrate.cuh"
#include "symbol.cuh"

int main() {
    std::cout << "Creating an expression" << std::endl;
    std::vector<std::vector<Sym::Symbol>> expressions = {Sym::cos(Sym::var()),
                                                         Sym::sin(Sym::cos(Sym::var())),
                                                         Sym::e() ^ Sym::var(),
                                                         Sym::var() ^ Sym::num(5),
                                                         Sym::var() ^ Sym::pi(),
                                                         Sym::var() ^ Sym::var(),
                                                         Sym::pi()};
    std::cout << "Expression created" << std::endl;

    for (size_t i = 0; i < expressions.size(); ++i) {
        std::cout << expressions[i][0].to_string() << std::endl;
    }

    std::cout << "Allocating and zeroing GPU memory" << std::endl;

    Sym::Symbol* d_expressions;
    cudaMalloc(&d_expressions, Sym::EXPRESSION_ARRAY_SIZE * sizeof(Sym::Symbol));
    cudaMemset(d_expressions, 0, Sym::EXPRESSION_ARRAY_SIZE * sizeof(Sym::Symbol));

    size_t* d_applicability;
    cudaMalloc(&d_applicability, Sym::APPLICABILITY_ARRAY_SIZE * sizeof(size_t));
    cudaMemset(d_applicability, 0, Sym::APPLICABILITY_ARRAY_SIZE * sizeof(size_t));

    std::cout << "Allocated GPU memory" << std::endl;
    std::cout << "Copying to GPU memory" << std::endl;

    for (size_t i = 0; i < expressions.size(); ++i) {
        cudaMemcpy(d_expressions + Sym::EXPRESSION_MAX_SYMBOL_COUNT * i, expressions[i].data(),
                   expressions[i].size() * sizeof(Sym::Symbol), cudaMemcpyHostToDevice);
    }

    std::cout << "Copied to GPU memory" << std::endl;
    std::cout << "Checking heuristics" << std::endl;

    Sym::check_heuristics_applicability<<<32, 1024>>>(d_expressions, d_applicability,
                                                      expressions.size());

    std::cout << "Checked heuristics" << std::endl;
    std::cout << "Calculating partial sum of applicability" << std::endl;

    thrust::inclusive_scan(thrust::device, d_applicability,
                           d_applicability + Sym::APPLICABILITY_ARRAY_SIZE, d_applicability);

    std::cout << "Calculated partial sum of applicability" << std::endl;
    std::cout << "Copying results to host memory" << std::endl;

    std::vector<size_t> h_applicability(expressions.size());
    h_applicability.resize(Sym::APPLICABILITY_ARRAY_SIZE);
    cudaMemcpy(h_applicability.data(), d_applicability,
               Sym::APPLICABILITY_ARRAY_SIZE * sizeof(size_t), cudaMemcpyDeviceToHost);

    std::cout << "Copied results to host memory" << std::endl;

    for (size_t i = 0; i < h_applicability.size(); ++i) {
        if (i % Sym::MAX_EXPRESSION_COUNT == 0) {
            std::cout << std::endl;
        }

        std::cout << h_applicability[i] << ", ";
    }
    std::cout << std::endl;

    std::cout << "Freeing GPU memory" << std::endl;
    cudaFree(d_applicability);
    cudaFree(d_expressions);
    std::cout << "Freed GPU memory" << std::endl;
}
