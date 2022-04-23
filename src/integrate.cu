#include "integrate.cuh"

namespace Sym {
    __device__ HeuristicCheck heuristic_checks[HEURISTIC_CHECK_COUNT] = {
        is_simple_variable_power, is_variable_exponent, is_simple_sine, is_simple_cosine, is_sum};

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

    __device__ size_t is_sum(Symbol* expression) { return expression[0].is(Type::Addition); }

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
} // namespace Sym
