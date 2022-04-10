#include "integrate.cuh"

namespace Sym {
    __device__ HeuristicCheck heuristic_checks[HEURISTIC_CHECK_COUNT] = {
        is_simple_variable_power, is_variable_exponent, is_simple_sine, is_simple_cosine, is_sum};

    __device__ bool is_simple_variable_power(Symbol* expression) {
        return expression[0].is(Type::Power) && expression[1].is(Type::Variable) &&
               (expression[2].is(Type::NumericConstant) &&
                    expression[2].numeric_constant.value != 0.0 ||
                expression[2].is(Type::KnownConstant) || expression[2].is(Type::UnknownConstant));
    }
    __device__ bool is_variable_exponent(Symbol* expression) {
        return expression[0].is(Type::Power) && expression[1].is(Type::KnownConstant) &&
               expression[1].known_constant.value == KnownConstantValue::E &&
               expression[2].is(Type::Variable);
    }
    __device__ bool is_simple_sine(Symbol* expression) {
        return expression[0].is(Type::Sine) && expression[1].is(Type::Variable);
    }

    __device__ bool is_simple_cosine(Symbol* expression) {
        return expression[0].is(Type::Cosine) && expression[1].is(Type::Variable);
    }

    __device__ bool is_sum(Symbol* expression) { return expression[0].is(Type::Addition); }

    // TODO: Do not assume that the number of threads is sufficient for the number of expressions
    // and heuristics to run everything in parallel
    __global__ void check_heuristics_applicability(Symbol** expressions, bool** applicability,
                                                   size_t expression_count) {
        size_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
        size_t heuristic_group_idx = thread_idx / HEURISITC_GROUP_SIZE;
        if (heuristic_group_idx >= HEURISTIC_CHECK_COUNT) {
            return;
        }

        size_t expression_idx = thread_idx % HEURISITC_GROUP_SIZE;
        if (expression_idx >= expression_count) {
            return;
        }

        applicability[expression_idx][heuristic_group_idx] =
            heuristic_checks[heuristic_group_idx](expressions[expression_idx]);
    }
} // namespace Sym
