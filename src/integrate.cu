#include "integrate.cuh"

namespace Sym {
    __device__ HeuristicCheck heuristic_checks[] = {is_simple_variable_power, is_variable_exponent,
                                                    is_simple_sine, is_simple_cosine};

    __device__ HeuristicApplication heuristic_applications[] = {
        transform_simple_variable_power, transform_variable_exponent, transform_simple_sine,
        transform_simple_cosine};

    static_assert(sizeof(heuristic_applications) == sizeof(heuristic_checks),
                  "Different number of heuristics and applications defined");

    static_assert(sizeof(heuristic_checks) == sizeof(HeuristicCheck) * HEURISTIC_CHECK_COUNT,
                  "HEURISTIC_CHECK_COUNT is not equal to number of heuristic checks");

    __device__ size_t is_simple_variable_power(Symbol* expression) {
        return expression[0].is(Type::Power) && expression[1].is(Type::Variable) &&
               (expression[2].is(Type::NumericConstant) &&
                    expression[2].numeric_constant.value != 0.0 ||
                expression[2].is(Type::KnownConstant) || expression[2].is(Type::UnknownConstant));
    }
    __device__ size_t is_variable_exponent(Symbol* expression) {
        return expression[0].is(Type::Power) && expression[1].is(Type::KnownConstant) &&
               expression[1].known_constant.value == KnownConstantValue::E &&
               expression[2].is(Type::Variable);
    }
    __device__ size_t is_simple_sine(Symbol* expression) {
        return expression[0].is(Type::Sine) && expression[1].is(Type::Variable);
    }

    __device__ size_t is_simple_cosine(Symbol* expression) {
        return expression[0].is(Type::Cosine) && expression[1].is(Type::Variable);
    }

    __device__ void transform_simple_variable_power(Symbol* expression, Symbol* destination) {
        destination[0].product = Product::create();
        destination[0].product.second_arg_offset = 2;
        destination[0].product.total_size = 7;

        destination[1] = expression[2]; // copy exponent

        destination[2].power = Power::create();
        destination[2].power.second_arg_offset = 2;
        destination[2].power.total_size = 5;

        destination[3] = expression[1]; // copy variable

        destination[4].addition = Addition::create();
        destination[4].addition.second_arg_offset = 2;
        destination[4].addition.total_size = 3;

        destination[5] = expression[2]; // copy exponent

        destination[6].numeric_constant = NumericConstant::create();
        destination[6].numeric_constant.value = -1.0;
    }

    __device__ void transform_variable_exponent(Symbol* expression, Symbol* destination) {
        destination[0] = expression[0]; // power
        destination[1] = expression[1]; // e constant
        destination[2] = expression[2]; // variable
    }

    __device__ void transform_simple_sine(Symbol* expression, Symbol* destination) {
        destination[0].negative = Negative::create();
        destination[0].negative.total_size = 3;

        destination[1].cosine = Cosine::create();
        destination[1].cosine.total_size = 2;

        destination[2] = expression[1]; // copy variable
    }

    __device__ void transform_simple_cosine(Symbol* expression, Symbol* destination) {
        destination[0].sine = Sine::create();
        destination[0].sine.total_size = 2;

        destination[1] = expression[1]; // copy variable
    }

    __global__ void check_heuristics_applicability(Symbol* expressions, size_t* applicability,
                                                   size_t expression_count) {
        size_t thread_count = gridDim.x * blockDim.x;
        size_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x;

        size_t hrstc_step = thread_count / HEURISITC_GROUP_SIZE;

        for (size_t hrstc_idx = thread_idx / HEURISITC_GROUP_SIZE;
             hrstc_idx < HEURISTIC_CHECK_COUNT; hrstc_idx += hrstc_step) {
            for (size_t expr_idx = thread_idx % HEURISITC_GROUP_SIZE; expr_idx < expression_count;
                 expr_idx += HEURISITC_GROUP_SIZE) {
                Symbol* expression_pointer = expressions + expr_idx * EXPRESSION_MAX_SYMBOL_COUNT;
                size_t applicability_index = MAX_EXPRESSION_COUNT * hrstc_idx + expr_idx;
                applicability[applicability_index] =
                    heuristic_checks[hrstc_idx](expression_pointer);
            }
        }
    }

    __global__ void apply_heuristics(Symbol* expressions, Symbol* destination,
                                     size_t* applicability, size_t expression_count) {
        size_t thread_count = gridDim.x * blockDim.x;
        size_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x;

        size_t hrstc_step = thread_count / HEURISITC_GROUP_SIZE;

        for (size_t hrstc_idx = thread_idx / HEURISITC_GROUP_SIZE;
             hrstc_idx < HEURISTIC_CHECK_COUNT; hrstc_idx += hrstc_step) {
            for (size_t expr_idx = thread_idx % HEURISITC_GROUP_SIZE; expr_idx < expression_count;
                 expr_idx += HEURISITC_GROUP_SIZE) {
                Symbol* expression_pointer = expressions + expr_idx * EXPRESSION_MAX_SYMBOL_COUNT;
                size_t applicability_index = MAX_EXPRESSION_COUNT * hrstc_idx + expr_idx;

                if (applicability_index == 0 && applicability[applicability_index] != 0 ||
                    applicability_index != 0 && applicability[applicability_index - 1] !=
                                                    applicability[applicability_index]) {
                    size_t destination_offset =
                        EXPRESSION_MAX_SYMBOL_COUNT * (applicability[applicability_index] - 1);
                    Symbol* destination_pointer = destination + destination_offset;
                    heuristic_applications[hrstc_idx](expression_pointer, destination_pointer);
                }
            }
        }
    }
}
