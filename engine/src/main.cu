#include <cstdlib>
#include <cstring>

#include <iostream>
#include <optional>
#include <vector>

#include <fmt/core.h>

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "Evaluation/Integrate.cuh"

#include "Symbol/Integral.cuh"
#include "Symbol/Symbol.cuh"

#include "Utils/CompileConstants.cuh"

static constexpr size_t BLOCK_SIZE = 512;
static constexpr size_t BLOCK_COUNT = 32;

std::optional<std::vector<std::vector<Sym::Symbol>>>
solve_integral(const std::vector<Sym::Symbol>& integral) {
    Sym::ExpressionArray<> expressions({Sym::single_integral_vacancy()},
                                       Sym::EXPRESSION_MAX_SYMBOL_COUNT, Sym::MAX_EXPRESSION_COUNT);
    Sym::ExpressionArray<> expressions_swap(Sym::MAX_EXPRESSION_COUNT,
                                            Sym::EXPRESSION_MAX_SYMBOL_COUNT, expressions.size());

    Sym::ExpressionArray<Sym::SubexpressionCandidate> integrals(
        {Sym::first_expression_candidate(integral)}, Sym::MAX_EXPRESSION_COUNT,
        Sym::EXPRESSION_MAX_SYMBOL_COUNT);
    Sym::ExpressionArray<Sym::SubexpressionCandidate> integrals_swap(
        Sym::MAX_EXPRESSION_COUNT, Sym::EXPRESSION_MAX_SYMBOL_COUNT);
    Sym::ExpressionArray<> help_spaces(Sym::MAX_EXPRESSION_COUNT, Sym::EXPRESSION_MAX_SYMBOL_COUNT,
                                       integrals.size());
    Util::DeviceArray<uint32_t> scan_array_1(Sym::SCAN_ARRAY_SIZE, true);
    Util::DeviceArray<uint32_t> scan_array_2(Sym::SCAN_ARRAY_SIZE, true);

    for (;;) {
        Sym::simplify<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, help_spaces);
        cudaDeviceSynchronize();

        Sym::check_for_known_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, scan_array_1);
        cudaDeviceSynchronize();

        thrust::inclusive_scan(thrust::device, scan_array_1.begin(), scan_array_1.end(),
                               scan_array_1.data());
        cudaDeviceSynchronize();

        Sym::apply_known_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, expressions, help_spaces,
                                                                scan_array_1);
        cudaDeviceSynchronize();
        expressions.increment_size_from_device(scan_array_1.last());

        Sym::propagate_solved_subexpressions<<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions);
        cudaDeviceSynchronize();

        std::vector<Sym::Symbol> first_expression = expressions.to_vector(0);
        if (first_expression.data()->as<Sym::SubexpressionVacancy>().is_solved == 1) {
            // TODO: Collapse the tree instead of returning it
            Sym::simplify<<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions, help_spaces);
            return expressions.to_vector();
        }

        scan_array_1.set_mem(1); // TODO: Causes problems when .last() is used later, should be ones
                                 // only for expressions, zeros elsewhere
        Sym::find_redundand_expressions<<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions, scan_array_1);
        cudaDeviceSynchronize();

        scan_array_2.zero_mem();
        Sym::find_redundand_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, expressions,
                                                                   scan_array_1, scan_array_2);
        cudaDeviceSynchronize();

        thrust::inclusive_scan(thrust::device, scan_array_1.begin(), scan_array_1.end(),
                               scan_array_1.data());
        thrust::inclusive_scan(thrust::device, scan_array_2.begin(), scan_array_2.end(),
                               scan_array_2.data());
        cudaDeviceSynchronize();

        Sym::remove_expressions<true>
            <<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions, scan_array_1, expressions_swap);
        Sym::remove_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, scan_array_2, scan_array_1,
                                                           integrals_swap);
        cudaDeviceSynchronize();

        std::swap(expressions, expressions_swap);
        std::swap(integrals, integrals_swap);
        expressions.resize_from_device(scan_array_1.last());
        integrals.resize_from_device(scan_array_2.last());

        scan_array_1.zero_mem();
        scan_array_2.zero_mem();
        Sym::check_heuristics_applicability<<<BLOCK_COUNT, BLOCK_SIZE>>>(
            integrals, expressions, scan_array_1, scan_array_2);
        cudaDeviceSynchronize();

        thrust::inclusive_scan(thrust::device, scan_array_1.begin(), scan_array_1.end(),
                               scan_array_1.data());
        thrust::inclusive_scan(thrust::device, scan_array_2.begin(), scan_array_2.end(),
                               scan_array_2.data());
        cudaDeviceSynchronize();

        Sym::apply_heuristics<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, integrals_swap, expressions,
                                                           help_spaces, scan_array_1, scan_array_2);
        cudaDeviceSynchronize();

        std::swap(integrals, integrals_swap);
        integrals.resize_from_device(scan_array_1.last());
        expressions.increment_size_from_device(scan_array_2.last());

        scan_array_1.set_mem(1); // TODO: Causes problems when .last() is used later, should be ones
                                 // only for expressions, zeros elsewhere
        cudaDeviceSynchronize();

        Sym::propagate_failures_upwards<<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions, scan_array_1);
        cudaDeviceSynchronize();

        // First expression in the array has failed, all is lost
        if (scan_array_1.to_cpu(0) == 0) {
            return std::nullopt;
        }

        Sym::propagate_failures_downwards<<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions, scan_array_1);
        cudaDeviceSynchronize();

        scan_array_2.zero_mem();
        Sym::find_redundand_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, scan_array_1,
                                                                   scan_array_2);
        cudaDeviceSynchronize();

        thrust::inclusive_scan(thrust::device, scan_array_1.begin(), scan_array_1.end(),
                               scan_array_1.data());
        thrust::inclusive_scan(thrust::device, scan_array_2.begin(), scan_array_2.end(),
                               scan_array_2.data());
        cudaDeviceSynchronize();

        Sym::remove_expressions<false>
            <<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions, scan_array_1, expressions_swap);
        Sym::remove_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, scan_array_2, scan_array_1,
                                                           integrals_swap);
        cudaDeviceSynchronize();

        std::swap(expressions, expressions_swap);
        std::swap(integrals, integrals_swap);
        expressions.resize_from_device(scan_array_1.last());
        integrals.resize_from_device(scan_array_2.last());
    }

    return std::nullopt;
}

int main() {
    if constexpr (Consts::DEBUG) {
        fmt::print("Running in debug mode\n");
    }

    std::vector<Sym::Symbol> integral =
        /* Sym::integral((Sym::e() ^ Sym::var()) * (Sym::e() ^ (Sym::e() ^ Sym::var()))); */
        Sym::integral(Sym::var() ^ (Sym::e() + Sym::num(10) + Sym::pi() +
                                    Sym::cnst("phi") * Sym::num(2) + Sym::num(5)));

    fmt::print("Trying to solve an integral: {}\n", integral.data()->to_string());

    std::optional<std::vector<std::vector<Sym::Symbol>>> solution = solve_integral(integral);

    if (solution.has_value()) {
        fmt::print("Success! Expressions tree:\n");
        for (const auto& expr : solution.value()) {
            fmt::print("{}\n", expr.data()->to_string());
        }
    }
    else {
        fmt::print("No solution found\n");
    }
}
