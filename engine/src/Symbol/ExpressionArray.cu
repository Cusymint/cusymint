#include "ExpressionArray.cuh"
#include "Utils/Cuda.cuh"

namespace Sym::ExpressionArrayKernel {
    __global__ void set_new_expression_capacities(
        Util::DeviceArray<size_t> expression_capacities,
        Util::DeviceArray<size_t> expression_capacities_sum, const size_t old_expression_count,
        const size_t new_expression_count, const size_t new_expressions_capacity) {
        const size_t thread_idx = Util::thread_idx();
        const size_t thread_count = Util::thread_count();

        const size_t size_diff = new_expression_count - old_expression_count;
        const size_t old_expressions_total_capacity =
            old_expression_count == 0 ? 0 : expression_capacities_sum[old_expression_count - 1];

        for (size_t i = thread_idx; i < size_diff; i += thread_count) {
            expression_capacities[i + old_expression_count] = new_expressions_capacity;
            expression_capacities_sum[i + old_expression_count] =
                old_expressions_total_capacity + new_expressions_capacity * (i + 1);
        }
    }

    __global__ void multiply_capacities(const Util::DeviceArray<EvaluationStatus> statuses,
                                        Util::DeviceArray<size_t> expression_capacities,
                                        const size_t realloc_multiplier,
                                        const size_t expression_count, const size_t start) {
        const size_t thread_idx = Util::thread_idx();
        const size_t thread_count = Util::thread_count();

        for (size_t i = thread_idx; i < expression_count - start; i += thread_count) {
            const size_t one_zero = statuses[i] == EvaluationStatus::ReallocationRequest ? 1 : 0;
            // Multiplier equal to `realloc_multiplier` when `indices[i]` is `true`, equal to `1`
            // otherwise
            expression_capacities[i + start] =
                (one_zero * (realloc_multiplier - 1) + 1) * expression_capacities[i + start];
        }
    }

    __global__ void repeat_capacities(Util::DeviceArray<size_t> expression_capacities,
                                      Util::DeviceArray<size_t> expression_capacities_sum,
                                      const size_t original_expression_count,
                                      const size_t original_total_size) {
        const size_t thread_idx = Util::thread_idx();
        const size_t thread_count = Util::thread_count();

        if (original_expression_count == 1) {
            for (size_t expr_idx = thread_idx; expr_idx < expression_capacities.size();
                 expr_idx += thread_count) {
                expression_capacities[expr_idx] = original_total_size;
                expression_capacities_sum[expr_idx] = original_total_size * (expr_idx + 1);
            }
            return;
        }

        for (size_t expr_idx = thread_idx + original_expression_count;
             expr_idx <= expression_capacities.size(); expr_idx += thread_count) {
            const size_t repeat_offset =
                (expr_idx / original_expression_count) * original_total_size;
            const size_t capacity_index = expr_idx % original_expression_count;
            expression_capacities_sum[expr_idx - 1] = repeat_offset;
            if (capacity_index > 0) {
                expression_capacities_sum[expr_idx - 1] +=
                    expression_capacities_sum[capacity_index - 1];
                expression_capacities[expr_idx - 1] = expression_capacities[capacity_index - 1];
            }
            else {
                expression_capacities[expr_idx - 1] =
                    original_total_size - expression_capacities_sum[original_expression_count - 2];
            }
        }
    }

    __global__ void reoffset_data(const Util::DeviceArray<Symbol> old_data,
                                  Util::DeviceArray<Symbol> new_data,
                                  const Util::DeviceArray<size_t> old_expression_capacities_sum,
                                  const Util::DeviceArray<size_t> new_expression_capacities_sum,
                                  const size_t expression_count) {
        const size_t thread_idx = Util::thread_idx();
        const size_t thread_count = Util::thread_count();

        for (size_t expr_idx = thread_idx; expr_idx < expression_count; expr_idx += thread_count) {
            const size_t old_data_idx =
                expr_idx == 0 ? 0 : old_expression_capacities_sum[expr_idx - 1];
            const size_t new_data_idx =
                expr_idx == 0 ? 0 : new_expression_capacities_sum[expr_idx - 1];
            const size_t expr_capacity = old_expression_capacities_sum[expr_idx] - old_data_idx;

            for (size_t sym_idx = 0; sym_idx < expr_capacity; ++sym_idx) {
                new_data[new_data_idx + sym_idx] = old_data[old_data_idx + sym_idx];
            }
        }
    }

    __global__ void reoffset_scan(const Util::DeviceArray<size_t> other_capacities,
                                  const Util::DeviceArray<uint32_t> scan,
                                  Util::DeviceArray<size_t> capacities, const size_t other_size) {
        const size_t thread_idx = Util::thread_idx();
        const size_t thread_count = Util::thread_count();

        for (size_t i = thread_idx; i < other_size; i += thread_count) {
            const bool is_masked = i == 0 && scan[i] != 0 || i != 0 && scan[i - 1] != scan[i];

            if (!is_masked) {
                continue;
            }

            const size_t dst_idx = scan[i] - 1;
            capacities[dst_idx] = other_capacities[i];
        }
    }

    __global__ void multiply_capacities(Util::DeviceArray<size_t> capacities,
                                        Util::DeviceArray<size_t> capacities_sum,
                                        const size_t multiplier) {

        const size_t thread_idx = Util::thread_idx();
        const size_t thread_count = Util::thread_count();

        for (size_t i = thread_idx; i < capacities.size(); i += thread_count) {
            capacities[i] = multiplier * capacities[i];
            capacities_sum[i] = multiplier * capacities_sum[i];
        }
    }
}
