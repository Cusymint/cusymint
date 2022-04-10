#include <cstdlib>
#include <cstring>

#include <iostream>
#include <vector>

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

    for(size_t i = 0; i < expressions.size(); ++i) {
        std::cout << expressions[i][0].to_string() << std::endl;
    }

    std::cout << "Allocating GPU memory" << std::endl;

    std::vector<Sym::Symbol*> h_d_expressions(expressions.size());
    std::vector<bool*> h_d_applicability(expressions.size());

    for (size_t i = 0; i < expressions.size(); ++i) {
        cudaMalloc(&h_d_expressions[i], expressions[i].size() * sizeof(Sym::Symbol));
        cudaMalloc(&h_d_applicability[i], Sym::HEURISTIC_CHECK_COUNT * sizeof(bool));
    }

    Sym::Symbol** d_expressions;
    cudaMalloc(&d_expressions, expressions.size() * sizeof(Sym::Symbol*));

    bool** d_applicability;
    cudaMalloc(&d_applicability, expressions.size() * sizeof(bool*));

    std::cout << "Allocated GPU memory" << std::endl;

    std::cout << "Copying to GPU memory" << std::endl;

    for (size_t i = 0; i < expressions.size(); ++i) {
        cudaMemcpy(h_d_expressions[i], expressions[i].data(),
                   expressions[i].size() * sizeof(Sym::Symbol), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(d_expressions, h_d_expressions.data(), h_d_expressions.size() * sizeof(Sym::Symbol*),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_applicability, h_d_applicability.data(), h_d_applicability.size() * sizeof(bool*),
               cudaMemcpyHostToDevice);

    std::cout << "Copied to GPU memory" << std::endl;

    std::cout << "Checking heuristics" << std::endl;

    Sym::check_heuristics_applicability<<<32, 1024>>>(d_expressions, d_applicability,
                                                      expressions.size());

    std::cout << "Checked heuristics" << std::endl;

    std::cout << "Copying results to host memory" << std::endl;

    // std::vector<bool> is specialised and may actually be a bit array
    std::vector<std::vector<uint8_t>> applicability(expressions.size());
    static_assert(sizeof(uint8_t) == sizeof(bool), "type is other size than bool");

    for (size_t i = 0; i < applicability.size(); ++i) {
        applicability[i].resize(Sym::HEURISTIC_CHECK_COUNT);
        cudaMemcpy(applicability[i].data(), h_d_applicability[i],
                   sizeof(bool) * Sym::HEURISTIC_CHECK_COUNT, cudaMemcpyDeviceToHost);
    }

    std::cout << "Copied results to host memory" << std::endl;

    for (size_t i = 0; i < applicability.size(); ++i) {
        std::cout << std::endl;
        for (size_t j = 0; j < applicability[i].size(); ++j) {
            std::cout << static_cast<bool>(applicability[i][j]) << ", ";
        }
    }

    std::cout << std::endl;

    std::cout << "Freeing GPU memory" << std::endl;
    for (size_t i = 0; i < expressions.size(); ++i) {
        cudaFree(h_d_expressions[i]);
    }
    cudaFree(d_expressions);
    std::cout << "Freed GPU memory" << std::endl;
}
