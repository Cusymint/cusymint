#ifndef SYMBOL_CUH
#define SYMBOL_CUH

#include <string>
#include <vector>

#include "addition.cuh"
#include "constants.cuh"
#include "integral.cuh"
#include "power.cuh"
#include "product.cuh"
#include "solution.cuh"
#include "substitution.cuh"
#include "trigonometric.cuh"
#include "unknown.cuh"
#include "variable.cuh"

namespace Sym {
    union Symbol {
        Unknown unknown;
        Variable variable;
        NumericConstant numeric_constant;
        KnownConstant known_constant;
        UnknownConstant unknown_constant;
        Integral integral;
        Solution solution;
        Substitution substitution;
        Addition addition;
        Negative negative;
        Product product;
        Reciprocal reciprocal;
        Power power;
        Sine sine;
        Cosine cosine;
        Tangent tangent;
        Cotangent cotangent;

        __host__ __device__ inline Type type() const { return unknown.type; }
        __host__ __device__ inline bool is(Type type) const { return unknown.type == type; }
        __host__ __device__ inline size_t total_size() const { return unknown.total_size; }

        template <class T> __host__ __device__ inline T& as() {
            return *reinterpret_cast<T*>(this);
        }

        template <class T> __host__ __device__ inline const T& as() const {
            return *reinterpret_cast<const T*>(this);
        }

        template <class T> __host__ __device__ static inline Symbol* from(T* sym) {
            return reinterpret_cast<Symbol*>(sym);
        }

        template <class T> __host__ __device__ static inline const Symbol* from(const T* sym) {
            return reinterpret_cast<const Symbol*>(sym);
        }

        __host__ __device__ inline const Symbol* child() const { return this + 1; }
        __host__ __device__ inline Symbol* child() { return this + 1; }

        /*
         * @brief Copies symbol sequence from `src` to `dst`.
         *
         * @param seq Symbol sequence to copy. Doesn't need to be semantically correct.
         * @param n number of symbols to copy
         */
        __host__ __device__ static void copy_symbol_sequence(Symbol* const dst,
                                                             const Symbol* const src, size_t n);

        /*
         * @brief Copies symbol sequence from `src` to `dst` in reverse.
         *
         * @param seq Symbol sequence to copy. Doesn't need to be semantically correct.
         * @param n number of symbols to copy
         */
        __host__ __device__ static void
        copy_and_reverse_symbol_sequence(Symbol* const dst, const Symbol* const src, size_t n);

        /*
         * @brief Checks if first n symbols of `seq1` and `seq2` are identical.
         * `seq1` and `seq2` don't need to be semantically correct
         *
         * @param seq1 First symbol sequence
         * @param seq2 Second symbol sequence
         * @param n number of symbols to compare
         *
         * @return `true` if they are the same, `false` otherwise
         */
        __host__ __device__ static bool are_symbol_sequences_same(const Symbol* seq1,
                                                                  const Symbol* seq2, size_t n);

        /*
         * @brief Swap contents of two symbols
         *
         * @param s1 First symbol
         * @param s2 Second symbol
         */
        __host__ __device__ static void swap_symbols(Symbol* const s1, Symbol* const s2);

        /*
         * @brief Reverses a symbol sequence in place.
         *
         * @param seq Symbol sequence to reverse. Doesn't need to be semantically correct.
         * @param n Sequence length
         */
        __host__ __device__ static void reverse_symbol_sequence(Symbol* const seq, size_t n);

        /*
         * @brief Copies `*this` into `dst`. Does not copy the whole tree, only a single symbol.
         *
         * @param dst Copy destination
         */
        __host__ __device__ void copy_single_to(Symbol* const dst) const;

        /*
         * @brief Copies `this` into `dst`. Copies the whole tree.
         *
         * @param dst Copy destination
         */
        __host__ __device__ void copy_to(Symbol* const dst) const;

        /*
         * @brief Checks if `this` is an expression composed only of constants
         *
         * @return `true` if `this` has no variables, `false` otherwise
         */
        __host__ __device__ bool is_constant() const;

        /*
         * @brief Returns offset of first occurence of variable in this symbol sequence
         *
         * @return Offset of first variable symbol. If there is none, then -1.
         */
        __host__ __device__ ssize_t first_var_occurence() const;

        /*
         * @brief Checks if variables in `this` are only part of expression `expression`
         *
         * @param expression Expression with variable to compare against every occurence of
         * variables in `this`.
         *
         * @return `true` if `this` is a function of `expression`, false otherwise. Returns false
         * also when `this` is constant. Although formally not correct, it isn't very usefull to
         * consider constant expressions as functions
         */
        __host__ __device__ bool is_function_of(Symbol* expression) const;

        /*
         * @brief Replaces every occurence of `expr` (which has to contain a variable) in `this`
         * with variable and copies the result to `destination`. If size of `expr` is larger than 1,
         * holes are left where symbols were before.
         *
         * @param destination Destination of copy
         * @param expr Expression to replace in `this`. Has to contain a variable.
         */
        __host__ __device__ void
        substitute_with_var_with_holes(Symbol* const destination,
                                       const Symbol* const expression) const;

        /*
         * @brief Removes holes from symbol tree and copies it in reverse order to `destination`.
         *
         * @param destination Location to which the tree is going to be copied
         *
         * @return New size of the symbol tree
         */
        __host__ __device__ size_t compress_reverse_to(Symbol* const destination) const;

        /*
         * @brief Returns integrand pointer if `this` is an integral. Undefined behavior otherwise.
         *
         * @return Pointer to integrand
         */
        __host__ __device__ inline Symbol* integrand() { return integral.integrand(); }

        /*
         * @brief Substitutes all occurences of variable with `symbol`
         *
         * @param symbol Symbol to substitute variables with, cannot have any children
         */
        void substitute_variable_with(const Symbol symbol);

        std::string to_string() const;
    };

    __host__ __device__ bool operator==(const Symbol& sym1, const Symbol& sym2);
    __host__ __device__ bool operator!=(const Symbol& sym1, const Symbol& sym2);

#define VIRTUAL_CALL(_instance, _member_function, ...)                         \
    (([&]() {                                                                  \
        switch ((_instance).unknown.type) {                                    \
        case Type::Variable:                                                   \
            return (_instance).variable._member_function(__VA_ARGS__);         \
        case Type::NumericConstant:                                            \
            return (_instance).numeric_constant._member_function(__VA_ARGS__); \
        case Type::KnownConstant:                                              \
            return (_instance).known_constant._member_function(__VA_ARGS__);   \
        case Type::UnknownConstant:                                            \
            return (_instance).unknown_constant._member_function(__VA_ARGS__); \
        case Type::Integral:                                                   \
            return (_instance).integral._member_function(__VA_ARGS__);         \
        case Type::Solution:                                                   \
            return (_instance).solution._member_function(__VA_ARGS__);         \
        case Type::Substitution:                                               \
            return (_instance).substitution._member_function(__VA_ARGS__);     \
        case Type::Addition:                                                   \
            return (_instance).addition._member_function(__VA_ARGS__);         \
        case Type::Negative:                                                   \
            return (_instance).negative._member_function(__VA_ARGS__);         \
        case Type::Product:                                                    \
            return (_instance).product._member_function(__VA_ARGS__);          \
        case Type::Reciprocal:                                                 \
            return (_instance).reciprocal._member_function(__VA_ARGS__);       \
        case Type::Power:                                                      \
            return (_instance).power._member_function(__VA_ARGS__);            \
        case Type::Sine:                                                       \
            return (_instance).sine._member_function(__VA_ARGS__);             \
        case Type::Cosine:                                                     \
            return (_instance).cosine._member_function(__VA_ARGS__);           \
        case Type::Tangent:                                                    \
            return (_instance).tangent._member_function(__VA_ARGS__);          \
        case Type::Cotangent:                                                  \
            return (_instance).cotangent._member_function(__VA_ARGS__);        \
        case Type::Unknown:                                                    \
        default:                                                               \
            return (_instance).unknown._member_function(__VA_ARGS__);          \
        }                                                                      \
    })())
}

#endif
