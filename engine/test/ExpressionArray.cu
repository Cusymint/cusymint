#include <gtest/gtest.h>
#include <vector>

#include "Symbol/ExpressionArray.cuh"
#include "Utils/DeviceArray.cuh"

namespace Test {
    TEST(ExpressionArrayKernel, RepeatCapacities) {
        Util::DeviceArray<size_t> capacities({3, 4, 2});
        Util::DeviceArray<size_t> capacities_sum({3, 7, 9});

        const size_t original_total_size = 15;
        const size_t original_expression_count = capacities.size() + 1;
        const size_t repeat_count = 3;

        Util::DeviceArray<size_t> new_capacities(original_expression_count * repeat_count - 1);
        Util::DeviceArray<size_t> new_capacities_sum(original_expression_count * repeat_count - 1);

        cudaMemcpy(new_capacities.data(), capacities.data(), capacities.size() * sizeof(size_t),
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_capacities_sum.data(), capacities_sum.data(),
                   capacities_sum.size() * sizeof(size_t), cudaMemcpyDeviceToDevice);

        std::vector<size_t> expected_capacities = {3, 4, 2, 6, 3, 4, 2, 6, 3, 4, 2};
        std::vector<size_t> expected_capacities_sum = {3, 7, 9, 15, 18, 22, 24, 30, 33, 37, 39};

        Sym::ExpressionArrayKernel::repeat_capacities<<<1024, 4>>>(
            new_capacities, new_capacities_sum, original_expression_count, original_total_size);

        EXPECT_EQ(new_capacities.to_vector(), expected_capacities);
        EXPECT_EQ(new_capacities_sum.to_vector(), expected_capacities_sum);
    }

    TEST(ExpressionArrayKernel, RepeatCapacitiesWithSingleExpression) {
        const size_t original_total_size = 7;
        const size_t original_expression_count = 1;
        const size_t repeat_count = 3;

        Util::DeviceArray<size_t> new_capacities(repeat_count - 1);
        Util::DeviceArray<size_t> new_capacities_sum(repeat_count - 1);

        std::vector<size_t> expected_capacities = {7, 7};
        std::vector<size_t> expected_capacities_sum = {7, 14};

        Sym::ExpressionArrayKernel::repeat_capacities<<<1024, 4>>>(
            new_capacities, new_capacities_sum, original_expression_count, original_total_size);

        EXPECT_EQ(new_capacities.to_vector(), expected_capacities);
        EXPECT_EQ(new_capacities_sum.to_vector(), expected_capacities_sum);
    }
}