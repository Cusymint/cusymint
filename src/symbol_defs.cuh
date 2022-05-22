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
        Negation,
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
        size_t size;                                                 \
        bool simplified;                                                   \
                                                                           \
        __host__ __device__ static _name builder() {                       \
            return {                                                       \
                .type = Sym::Type::_name,                                  \
                .simplified = _simple,                                     \
            };                                                             \
        }                                                                  \
                                                                           \
        __host__ __device__ void seal();                                   \
                                                                           \
        __host__ __device__ static _name create() {                        \
            return {                                                       \
                .type = Sym::Type::_name,                                  \
                .size = 1,                                           \
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
#define END_DECLARE_SYMBOL(_name)                                                             \
    }                                                                                         \
    ;                                                                                         \
    static_assert(std::is_pod<_name>::value, "Type '" #_name "' has to be POD, but is not!"); \
                                                                                              \
    __host__ __device__ _name* operator<<(Symbol* const destination, _name&& target);         \
    __host__ __device__ _name* operator<<(Symbol& destination, _name&& target);

#define DEFINE_NO_OP_SEAL(_name) \
    void _name::seal() {}

#define DEFINE_COMPARE(_name) COMPARE_HEADER(_name::compare)

#define BASE_COMPARE(_name)                                                             \
    symbol->as<_name>().type == type && symbol->as<_name>().size == size && \
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
        return size;                                   \
    }

#define DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(_name)                \
    COMPRESS_REVERSE_TO_HEADER(_name::compress_reverse_to) {             \
        size_t new_arg_size = arg().compress_reverse_to(destination);    \
        copy_single_to(destination + new_arg_size);                      \
        destination[new_arg_size].unknown.size = new_arg_size + 1; \
        return new_arg_size + 1;                                         \
    }

#define DEFINE_TWO_ARGUMENT_OP_COMPRESS_REVERSE_TO(_name)                               \
    COMPRESS_REVERSE_TO_HEADER(_name::compress_reverse_to) {                            \
        size_t new_arg2_size = arg2().compress_reverse_to(destination);                 \
        size_t new_arg1_size = arg1().compress_reverse_to(destination + new_arg2_size); \
                                                                                        \
        copy_single_to(destination + new_arg1_size + new_arg2_size);                    \
        destination[new_arg1_size + new_arg2_size].unknown.size =                 \
            new_arg1_size + new_arg2_size + 1;                                          \
        (destination + new_arg1_size + new_arg2_size)->as<_name>().second_arg_offset =  \
            new_arg1_size + 1;                                                          \
                                                                                        \
        return new_arg1_size + new_arg2_size + 1;                                       \
    }

#define DEFINE_UNSUPPORTED_COMPRESS_REVERSE_TO(_name)                    \
    COMPRESS_REVERSE_TO_HEADER(_name::compress_reverse_to) {             \
        printf("ERROR: compress_reverse_to used on unsupported type: "); \
        printf(#_name);                                                  \
        printf("\n");                                                    \
        /* Return -1 to crash the whole program as soon as possible */   \
        return static_cast<size_t>(-1);                                  \
    }

#define DEFINE_INTO_DESTINATION_OPERATOR(_name)                                        \
    __host__ __device__ _name* operator<<(Symbol* const destination, _name&& target) { \
        destination->as<_name>() = target;                                             \
        return &destination->as<_name>();                                              \
    }                                                                                  \
                                                                                       \
    __host__ __device__ _name* operator<<(Symbol& destination, _name&& target) {       \
        destination.as<_name>() = target;                                              \
        return &destination.as<_name>();                                               \
    }

#define ONE_ARGUMENT_OP_SYMBOL                     \
    __host__ __device__ const Symbol& arg() const; \
    __host__ __device__ Symbol& arg();             \
    __host__ __device__ static void create(const Symbol* const arg, Symbol* const destination);

#define DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(_name)                                                  \
    DEFINE_INTO_DESTINATION_OPERATOR(_name)                                                      \
    __host__ __device__ const Symbol& _name::arg() const { return Symbol::from(this)[1]; }       \
                                                                                                 \
    __host__ __device__ Symbol& _name::arg() { return Symbol::from(this)[1]; }                   \
                                                                                                 \
    __host__ __device__ void _name::seal() { size = 1 + arg().size(); }              \
                                                                                                 \
    __host__ __device__ void _name::create(const Symbol* const arg, Symbol* const destination) { \
        _name* const one_arg_op = destination << _name::builder();                               \
        arg->copy_to(&one_arg_op->arg());                                                        \
        one_arg_op->seal();                                                                      \
    }

#define TWO_ARGUMENT_OP_SYMBOL                                                                 \
    /* W 95% przypadków second_arg_offset == 1 + arg1().size(), ale nie zawsze. */      \
    /* Przykładowo w `compress_reverse_to` pierwszy argument może nie mieć poprawnej */     \
    /* struktury, a potrzebny jest tam offset do drugiego argumentu (implicite w arg2()) */    \
    size_t second_arg_offset;                                                                  \
    __host__ __device__ const Symbol& arg1() const;                                            \
    __host__ __device__ Symbol& arg1();                                                        \
    __host__ __device__ const Symbol& arg2() const;                                            \
    __host__ __device__ Symbol& arg2();                                                        \
    __host__ __device__ void seal_arg1();                                                      \
    __host__ __device__ static void create(const Symbol* const arg1, const Symbol* const arg2, \
                                           Symbol* const destination);

#define DEFINE_TWO_ARGUMENT_OP_FUNCTIONS(_name)                                                  \
    DEFINE_INTO_DESTINATION_OPERATOR(_name)                                                      \
                                                                                                 \
    __host__ __device__ const Symbol& _name::arg1() const { return Symbol::from(this)[1]; }      \
                                                                                                 \
    __host__ __device__ Symbol& _name::arg1() { return Symbol::from(this)[1]; }                  \
                                                                                                 \
    __host__ __device__ const Symbol& _name::arg2() const {                                      \
        return Symbol::from(this)[second_arg_offset];                                            \
    };                                                                                           \
                                                                                                 \
    __host__ __device__ Symbol& _name::arg2() { return Symbol::from(this)[second_arg_offset]; }; \
                                                                                                 \
    __host__ __device__ void _name::seal_arg1() { second_arg_offset = 1 + arg1().size(); } \
                                                                                                 \
    __host__ __device__ void _name::seal() {                                                     \
        size = 1 + arg1().size() + arg2().size();                              \
    }                                                                                            \
                                                                                                 \
    __host__ __device__ void _name::create(const Symbol* const arg1, const Symbol* const arg2,   \
                                           Symbol* const destination) {                          \
        _name* const two_arg_op = destination << _name::builder();                               \
        arg1->copy_to(&two_arg_op->arg1());                                                      \
        two_arg_op->seal_arg1();                                                                 \
        arg2->copy_to(&two_arg_op->arg2());                                                      \
        two_arg_op->seal();                                                                      \
    }
};

#endif
