#ifndef SYMBOL_DEFS_CUH
#define SYMBOL_DEFS_CUH

#include <cstring>

#include <string>
#include <type_traits>

#include "cuda_utils.cuh"

namespace Sym {
    union Symbol;

    enum class Type : size_t {
        // Basic types
        Unknown = 0,
        Variable,
        NumericConstant, // rational constants with given numeric value, e.g. 5, 1.345, 12.44
        KnownConstant,   // well known real constants, e.g. pi, e (Euler's number)
        UnknownConstant, // constants marked with letters, e.g. a, phi, delta
        // Meta structures
        Integral,
        Solution,
        Substitution,
        // Arithmetic
        Addition,
        Negative,
        Product,
        Reciprocal,
        Power,
        // Trigonometric functions
        Sine,
        Cosine,
        Tangent,
        Cotangent
    };

#define COMPRESS_REVERSE_TO_HEADER(_compress_reverse_to) \
    __host__ __device__ size_t _compress_reverse_to(Symbol* const destination) const

#define COMPARE_HEADER(_compare) __host__ __device__ bool _compare(const Symbol* symbol) const

#define DECLARE_SYMBOL(_name, _simple)                                     \
    struct _name {                                                         \
        Sym::Type type;                                                    \
        size_t total_size;                                                 \
        bool simplified;                                                   \
                                                                           \
        __host__ __device__ static _name create() {                        \
            return {                                                       \
                .type = Sym::Type::_name,                                  \
                .total_size = 1,                                           \
                .simplified = _simple,                                     \
            };                                                             \
        }                                                                  \
                                                                           \
        COMPARE_HEADER(compare);                                           \
                                                                           \
        __host__ __device__ void copy_single_to(Symbol* const dst) const { \
            Util::copy_mem(dst, this, sizeof(_name));                      \
        }                                                                  \
                                                                           \
        COMPRESS_REVERSE_TO_HEADER(compress_reverse_to);

// A struct is POD iff it is standard-layout and trivial.
// standard-layout is required to guarantee that all symbolic types have the `type` member
// at offset 0 in the memory layout (necessary for later use in Symbol union).
// trivial is necessary for copying symbolic structs
#define END_DECLARE_SYMBOL(_name) \
    }                             \
    ;                             \
    static_assert(std::is_pod<_name>::value, "Type '" #_name "' has to be POD, but is not!");

#define DEFINE_COMPARE(_name) COMPARE_HEADER(_name::compare)

#define BASE_COMPARE(_name)                                                             \
    symbol->as<_name>().type == type && symbol->as<_name>().total_size == total_size && \
        symbol->as<_name>().simplified == simplified

#define ONE_ARGUMENT_OP_COMPARE(_name) true

#define TWO_ARGUMENT_OP_COMPARE(_name) symbol->as<_name>().second_arg_offset == second_arg_offset

#define DEFINE_SIMPLE_COMPARE(_name) \
    DEFINE_COMPARE(_name) { return BASE_COMPARE(_name); }

#define DEFINE_SIMPLE_ONE_ARGUMETN_OP_COMPARE(_name) \
    DEFINE_COMPARE(_name) { return BASE_COMPARE(_name) && ONE_ARGUMENT_OP_COMPARE(_name); }

#define DEFINE_SIMPLE_TWO_ARGUMENT_OP_COMPARE(_name) \
    DEFINE_COMPARE(_name) { return BASE_COMPARE(_name) && TWO_ARGUMENT_OP_COMPARE(_name); }

#define DEFINE_TO_STRING(_str) \
    std::string to_string() const { return _str; }

#define DEFINE_SIMPLE_COMPRESS_REVERSE_TO(_name)             \
    COMPRESS_REVERSE_TO_HEADER(_name::compress_reverse_to) { \
        copy_single_to(destination);                         \
        return total_size;                                   \
    }

#define DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(_name)                \
    COMPRESS_REVERSE_TO_HEADER(_name::compress_reverse_to) {             \
        const Symbol* child = Symbol::from(this) + 1;                    \
        size_t new_arg_size = child->compress_reverse_to(destination);   \
        copy_single_to(destination + new_arg_size);                      \
        destination[new_arg_size].unknown.total_size = new_arg_size + 1; \
        return new_arg_size + 1;                                         \
    }

#define DEFINE_TWO_ARGUMENT_OP_COMPRESS_REVERSE_TO(_name)                              \
    COMPRESS_REVERSE_TO_HEADER(_name::compress_reverse_to) {                           \
        const Symbol* arg2 = Symbol::from(this) + second_arg_offset;                   \
        size_t new_arg2_size = arg2->compress_reverse_to(destination);                 \
                                                                                       \
        const Symbol* arg1 = Symbol::from(this) + 1;                                   \
        size_t new_arg1_size = arg1->compress_reverse_to(destination + new_arg2_size); \
                                                                                       \
        copy_single_to(destination + new_arg1_size + new_arg2_size);                   \
        destination[new_arg1_size + new_arg2_size].unknown.total_size =                \
            new_arg1_size + new_arg2_size + 1;                                         \
        (destination + new_arg1_size + new_arg2_size)->as<_name>().second_arg_offset = \
            new_arg1_size + 1;                                                         \
                                                                                       \
        return new_arg1_size + new_arg2_size + 1;                                      \
    }

#define DEFINE_UNSUPPORTED_COMPRESS_REVERSE_TO(_name)                    \
    COMPRESS_REVERSE_TO_HEADER(_name::compress_reverse_to) {             \
        printf("ERROR: compress_reverse_to used on unsupported type: "); \
        printf(#_name);                                                  \
        printf("\n");                                                    \
        /* Return -1 to crash the whole program as soon as possible */   \
        return static_cast<size_t>(-1);                                  \
    }

#define ONE_ARGUMENT_OP_SYMBOL

#define TWO_ARGUMENT_OP_SYMBOL size_t second_arg_offset;

};

#endif
