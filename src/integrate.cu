#include "integrate.cuh"

#include "cuda_utils.cuh"

namespace Sym {
    __device__ ApplicabilityCheck known_integral_checks[] = {is_simple_variable_power,
                                                             is_variable_exponent, is_simple_sine,
                                                             is_simple_cosine, is_constant};

    __device__ IntegralTransform known_integral_applications[] = {
        integrate_simple_variable_power, integrate_variable_exponent, integrate_simple_sine,
        integrate_simple_cosine, integrate_constant};

    static_assert(sizeof(known_integral_applications) == sizeof(known_integral_checks),
                  "Different number of heuristics and applications defined");

    static_assert(sizeof(known_integral_checks) ==
                      sizeof(ApplicabilityCheck) * KNOWN_INTEGRAL_COUNT,
                  "HEURISTIC_CHECK_COUNT is not equal to number of heuristic checks");

    __device__ size_t dummy_heuristic_check(Symbol*) { return 0; }
    __device__ void dummy_heuristic_transform(Symbol*, Symbol*) {}

    __device__ ApplicabilityCheck heuristic_checks[] = {dummy_heuristic_check};

    __device__ IntegralTransform heuristic_applications[] = {dummy_heuristic_transform};

    static_assert(sizeof(heuristic_checks) == sizeof(heuristic_applications),
                  "Different number of heuristics and applications defined");

    static_assert(sizeof(heuristic_checks) == sizeof(ApplicabilityCheck) * HEURISTIC_CHECK_COUNT,
                  "HEURISTIC_CHECK_COUNT is not equal to number of heuristic checks");

    __device__ size_t is_simple_variable_power(Symbol* integral) {
        return integral[0].is(Type::Power) && integral[1].is(Type::Variable) &&
               (integral[2].is(Type::NumericConstant) &&
                    integral[2].numeric_constant.value != 0.0 ||
                integral[2].is(Type::KnownConstant) || integral[2].is(Type::UnknownConstant));
    }
    __device__ size_t is_variable_exponent(Symbol* integral) {
        return integral[0].is(Type::Power) && integral[1].is(Type::KnownConstant) &&
               integral[1].known_constant.value == KnownConstantValue::E &&
               integral[2].is(Type::Variable);
    }
    __device__ size_t is_simple_sine(Symbol* integral) {
        return integral[0].is(Type::Sine) && integral[1].is(Type::Variable);
    }

    __device__ size_t is_simple_cosine(Symbol* integral) {
        return integral[0].is(Type::Cosine) && integral[1].is(Type::Variable);
    }

    __device__ size_t is_constant(Symbol* integral) {
        return integral[0].is(Type::NumericConstant) || integral[0].is(Type::KnownConstant) ||
               integral[0].is(Type::UnknownConstant);
    }

    // TODO: Sometimes results in "too many resources requested for launch" error when block size is
    // 1024?
    __device__ void integrate_simple_variable_power(Symbol* integral, Symbol* destination) {
        size_t exponent_size = integral[2].unknown.total_size;

        destination[0].product = Product::create();
        destination[0].product.second_arg_offset = 5;
        destination[0].product.total_size = 8 + 2 * exponent_size;

        destination[1].reciprocal = Reciprocal::create();
        destination[1].reciprocal.total_size = 3 + exponent_size;

        destination[2].addition = Addition::create();
        destination[2].addition.total_size = 2 + exponent_size;
        destination[2].addition.second_arg_offset = 2;

        // copy exponent
        for (size_t i = 0; i < exponent_size; ++i) {
            destination[3 + i] = integral[2 + i];
        }

        destination[3 + exponent_size].numeric_constant = NumericConstant::create();
        destination[3 + exponent_size].numeric_constant.value = 1.0;

        destination[4 + exponent_size].power = Power::create();
        destination[4 + exponent_size].power.second_arg_offset = 2;
        destination[4 + exponent_size].power.total_size = 4 + exponent_size;

        destination[5 + exponent_size] = integral[1]; // copy variable

        destination[6 + exponent_size].addition = Addition::create();
        destination[6 + exponent_size].addition.second_arg_offset = 2;
        destination[6 + exponent_size].addition.total_size = 2 + exponent_size;

        // copy exponent
        for (size_t i = 0; i < exponent_size; ++i) {
            destination[7 + exponent_size + i] = integral[2 + i];
        }

        destination[7 + exponent_size * 2].numeric_constant = NumericConstant::create();
        destination[7 + exponent_size * 2].numeric_constant.value = 1.0;
    }

    __device__ void integrate_variable_exponent(Symbol* integral, Symbol* destination) {
        destination[0] = integral[0]; // power
        destination[1] = integral[1]; // e constant
        destination[2] = integral[2]; // variable
    }

    __device__ void integrate_simple_sine(Symbol* integral, Symbol* destination) {
        destination[0].negative = Negative::create();
        destination[0].negative.total_size = 3;

        destination[1].cosine = Cosine::create();
        destination[1].cosine.total_size = 2;

        destination[2] = integral[1]; // copy variable
    }

    __device__ void integrate_simple_cosine(Symbol* integral, Symbol* destination) {
        destination[0].sine = Sine::create();
        destination[0].sine.total_size = 2;

        destination[1] = integral[1]; // copy variable
    }

    __device__ void integrate_constant(Symbol* integral, Symbol* destination) {
        destination[0].product = Product::create();
        destination[0].product.total_size = 3;
        destination[0].product.second_arg_offset = 2;
        destination[1].variable = Variable::create();
        destination[2] = integral[0]; // copy constant
    }

    __device__ void check_applicability(Symbol* integrals, size_t* applicability,
                                        size_t* integral_count, ApplicabilityCheck* checks,
                                        size_t check_count) {
        size_t thread_count = Util::thread_count();
        size_t thread_idx = Util::thread_idx();

        size_t check_step = thread_count / TRANSFORM_GROUP_SIZE;

        for (size_t check_idx = thread_idx / TRANSFORM_GROUP_SIZE; check_idx < check_count;
             check_idx += check_step) {
            for (size_t expr_idx = thread_idx % TRANSFORM_GROUP_SIZE; expr_idx < *integral_count;
                 expr_idx += TRANSFORM_GROUP_SIZE) {
                Symbol* integral_pointer = integrals + expr_idx * INTEGRAL_MAX_SYMBOL_COUNT;
                size_t applicability_index = MAX_INTEGRAL_COUNT * check_idx + expr_idx;

                applicability[applicability_index] = checks[check_idx](integral_pointer);
            }
        }
    }

    __device__ void apply_transforms(Symbol* integrals, Symbol* destinations,
                                     size_t* applicability, size_t* integral_count,
                                     IntegralTransform* transforms, size_t transform_count) {
        size_t thread_count = Util::thread_count();
        size_t thread_idx = Util::thread_idx();

        size_t trans_step = thread_count / TRANSFORM_GROUP_SIZE;

        for (size_t trans_idx = thread_idx / TRANSFORM_GROUP_SIZE; trans_idx < transform_count;
             trans_idx += trans_step) {
            for (size_t expr_idx = thread_idx % TRANSFORM_GROUP_SIZE; expr_idx < *integral_count;
                 expr_idx += TRANSFORM_GROUP_SIZE) {
                Symbol* integral_pointer = integrals + expr_idx * INTEGRAL_MAX_SYMBOL_COUNT;
                size_t applicability_index = MAX_INTEGRAL_COUNT * trans_idx + expr_idx;

                if (applicability_index == 0 && applicability[applicability_index] != 0 ||
                    applicability_index != 0 && applicability[applicability_index - 1] !=
                                                    applicability[applicability_index]) {
                    size_t destination_offset =
                        INTEGRAL_MAX_SYMBOL_COUNT * (applicability[applicability_index] - 1);
                    Symbol* destination = destinations + destination_offset;

                    transforms[trans_idx](integral_pointer, destination);
                }
            }
        }
    }

    __global__ void check_for_known_integrals(Symbol* integrals, size_t* applicability,
                                              size_t* integral_count) {
        check_applicability(integrals, applicability, integral_count, known_integral_checks,
                            KNOWN_INTEGRAL_COUNT);
    }

    __global__ void apply_known_integrals(Symbol* integrals, Symbol* destinations,
                                          size_t* applicability, size_t* integral_count) {
        apply_transforms(integrals, destinations, applicability, integral_count,
                         known_integral_applications, KNOWN_INTEGRAL_COUNT);
    }

    __global__ void check_heuristics_applicability(Symbol* integrals, size_t* applicability,
                                                   size_t* integral_count) {
        check_applicability(integrals, applicability, integral_count, heuristic_checks,
                            HEURISTIC_CHECK_COUNT);
    }

    __global__ void apply_heuristics(Symbol* integrals, Symbol* destinations,
                                     size_t* applicability, size_t* integral_count) {
        apply_transforms(integrals, destinations, applicability, integral_count,
                         heuristic_applications, HEURISTIC_CHECK_COUNT);
    }
}
