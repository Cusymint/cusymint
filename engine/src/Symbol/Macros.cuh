#ifndef SYMBOL_DEFS_CUH
#define SYMBOL_DEFS_CUH

#include <cstring>

#include <fmt/core.h>
#include <string>
#include <type_traits>

#include "SymbolType.cuh"
#include "Utils/Cuda.cuh"

namespace Sym {
    union Symbol;
}

#define COMPRESS_REVERSE_TO_HEADER(_compress_reverse_to) \
    __host__ __device__ size_t _compress_reverse_to(Symbol* const destination) const

#define COMPARE_HEADER(_compare) __host__ __device__ bool _compare(const Symbol* const symbol) const

#define SIMPLIFY_IN_PLACE_HEADER(_simplify_in_place) \
    __host__ __device__ void _simplify_in_place(Symbol* const help_space)

#define DECLARE_SYMBOL(_name, _simple)                                           \
    struct _name {                                                               \
        constexpr static Sym::Type TYPE = Sym::Type::_name;                      \
        Sym::Type type;                                                          \
        size_t size;                                                             \
        bool simplified;                                                         \
                                                                                 \
        __host__ __device__ static _name builder() {                             \
            _name symbol{};                                                      \
            symbol.type = Sym::Type::_name;                                      \
            symbol.simplified = _simple;                                         \
            return symbol;                                                       \
        }                                                                        \
                                                                                 \
        __host__ __device__ void seal();                                         \
                                                                                 \
        __host__ __device__ static _name create() {                              \
            return {                                                             \
                .type = Sym::Type::_name,                                        \
                .size = 1,                                                       \
                .simplified = _simple,                                           \
            };                                                                   \
        }                                                                        \
                                                                                 \
        __host__ __device__ void copy_single_to(Symbol* const dst) const {       \
            Util::copy_mem(dst, this, sizeof(_name));                            \
        }                                                                        \
        __host__ __device__ inline const Symbol* this_symbol() const {           \
            return reinterpret_cast<const Symbol*>(this);                        \
        }                                                                        \
                                                                                 \
        __host__ __device__ inline Symbol* this_symbol() {                       \
            return reinterpret_cast<Symbol*>(this);                              \
        }                                                                        \
                                                                                 \
        template <class T> __host__ __device__ inline const T* this_as() const { \
            return reinterpret_cast<const T*>(this);                             \
        }                                                                        \
                                                                                 \
        template <class T> __host__ __device__ inline T* this_as() {             \
            return reinterpret_cast<T*>(this);                                   \
        }                                                                        \
                                                                                 \
        COMPARE_HEADER(compare);                                                 \
        COMPRESS_REVERSE_TO_HEADER(compress_reverse_to);                         \
        SIMPLIFY_IN_PLACE_HEADER(simplify_in_place);

// Struktura jest POD w.t.w. gdy jest stanard-layout i trivial.
// standard-layout jest wymagany by zagwarantować, że wszystkie symbole mają pole `type` na offsecie
// 0 (potrzebne do użycia w unii `Symbol`).
// trivial jest potrzebny do kopiowania symboli operacjami takimi jak `memcpy`.
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

#define DEFINE_NO_OP_SIMPLIFY_IN_PLACE(_name) \
    SIMPLIFY_IN_PLACE_HEADER(_name::simplify_in_place) {}

#define DEFINE_SIMPLIFY_IN_PLACE(_name) SIMPLIFY_IN_PLACE_HEADER(_name::simplify_in_place)

#define BASE_COMPARE(_name)                                                 \
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

#define DEFINE_COMPRESS_REVERSE_TO(_name) COMPRESS_REVERSE_TO_HEADER(_name::compress_reverse_to)

#define DEFINE_SIMPLE_COMPRESS_REVERSE_TO(_name) \
    DEFINE_COMPRESS_REVERSE_TO(_name) {          \
        copy_single_to(destination);             \
        return size;                             \
    }

#define DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(_name)                   \
    DEFINE_COMPRESS_REVERSE_TO(_name) {                                     \
        const size_t new_arg_size = arg().compress_reverse_to(destination); \
        copy_single_to(destination + new_arg_size);                         \
        destination[new_arg_size].unknown.size = new_arg_size + 1;          \
        return new_arg_size + 1;                                            \
    }

#define DEFINE_TWO_ARGUMENT_OP_COMPRESS_REVERSE_TO(_name)                                     \
    DEFINE_COMPRESS_REVERSE_TO(_name) {                                                       \
        const size_t new_arg2_size = arg2().compress_reverse_to(destination);                 \
        const size_t new_arg1_size = arg1().compress_reverse_to(destination + new_arg2_size); \
                                                                                              \
        copy_single_to(destination + new_arg1_size + new_arg2_size);                          \
        destination[new_arg1_size + new_arg2_size].unknown.size =                             \
            new_arg1_size + new_arg2_size + 1;                                                \
        (destination + new_arg1_size + new_arg2_size)->as<_name>().second_arg_offset =        \
            new_arg1_size + 1;                                                                \
                                                                                              \
        return new_arg1_size + new_arg2_size + 1;                                             \
    }

#define DEFINE_UNSUPPORTED_COMPRESS_REVERSE_TO(_name)                                \
    DEFINE_COMPRESS_REVERSE_TO(_name) {                                              \
        printf("ERROR: compress_reverse_to used on unsupported type: %s\n", #_name); \
        Util::crash("");                                                             \
        return 0;                                                                    \
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
    __host__ __device__ void _name::seal() { size = 1 + arg().size(); }                          \
                                                                                                 \
    __host__ __device__ void _name::create(const Symbol* const arg, Symbol* const destination) { \
        _name* const one_arg_op = destination << _name::builder();                               \
        arg->copy_to(&one_arg_op->arg());                                                        \
        one_arg_op->seal();                                                                      \
    }

#define TWO_ARGUMENT_OP_SYMBOL                                                                 \
    /* W 95% przypadków second_arg_offset == 1 + arg1().size(), ale nie zawsze                \
     * Przykładowo w `compress_reverse_to` pierwszy argument może nie mieć poprawnej        \
     * struktury, a potrzebny jest tam offset do drugiego argumentu (implicite w arg2())       \
     */                                                                                        \
    size_t second_arg_offset;                                                                  \
    __host__ __device__ const Symbol& arg1() const;                                            \
    __host__ __device__ Symbol& arg1();                                                        \
    __host__ __device__ const Symbol& arg2() const;                                            \
    __host__ __device__ Symbol& arg2();                                                        \
    __host__ __device__ void seal_arg1();                                                      \
    __host__ __device__ void swap_args(Symbol* const help_space);                              \
    __host__ __device__ static void create(const Symbol* const arg1, const Symbol* const arg2, \
                                           Symbol* const destination);

#define TWO_ARGUMENT_COMMUTATIVE_OP_SYMBOL(_name)                                             \
    TWO_ARGUMENT_OP_SYMBOL                                                                    \
    /*                                                                                        \
     * @brief W uproszczonym drzewie operatora zwraca operację najniżej w drzewie           \
     *                                                                                        \
     * @return Wskaźnik do ostatniego operator. Jeśli `arg1()` nie jest tego samego typu co \
     * `*this`, to zwraca `this`                                                              \
     */                                                                                       \
    __host__ __device__ const _name* last_in_tree() const;                                    \
                                                                                              \
    /*                                                                                        \
     * @brief Przeładowanie bez `const`                                                      \
     */                                                                                       \
    __host__ __device__ _name* last_in_tree();                                                \
                                                                                              \
    /*                                                                                        \
     * @brief Uproszczenie struktury operatora `$` do drzewa w postaci:                       \
     *                 $                                                                      \
     *                / \                                                                     \
     *               $   e                                                                    \
     *              / \                                                                       \
     *             $   d                                                                      \
     *            / \                                                                         \
     *           $   c                                                                        \
     *          / \                                                                           \
     *         a   b                                                                          \
     *                                                                                        \
     * Zakładamy, że oba argumenty są w uproszczonej postaci                               \
     *                                                                                        \
     * @param help_space Pamięć pomocnicza                                                  \
     */                                                                                       \
    __host__ __device__ void simplify_structure(Symbol* const help_space);                    \
                                                                                              \
    /*                                                                                        \
     * @brief W drzewie o uproszczonej strukturze wyszukuje par upraszczalnych wyrażeń.     \
     */                                                                                       \
    __host__ __device__ void simplify_pairs();                                                \
                                                                                              \
    /*                                                                                        \
     * @brief Sprawdza, czy dwa drzewa można uprościć operatorem. Jeśli tak, to to robi   \
     *                                                                                        \
     * @param expr1 Pierwszy argument operatora                                               \
     * @param expr2 Drugi argument operatora                                                  \
     *                                                                                        \
     * @return `true` jeśli wykonano uproszczenie, `false` w przeciwnym wypadku              \
     */                                                                                       \
    __host__ __device__ static bool try_fuse_symbols(Symbol* const expr1, Symbol* const expr2);

#define DEFINE_TRY_FUSE_SYMBOLS(_name) \
    __host__ __device__ bool _name::try_fuse_symbols(Symbol* const expr1, Symbol* const expr2)

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
    __host__ __device__ void _name::seal_arg1() { second_arg_offset = 1 + arg1().size(); }       \
                                                                                                 \
    __host__ __device__ void _name::seal() { size = 1 + arg1().size() + arg2().size(); }         \
                                                                                                 \
    __host__ __device__ void _name::swap_args(Symbol* const help_space) {                        \
        arg1().copy_to(help_space);                                                              \
        arg2().copy_to(&arg1());                                                                 \
        seal_arg1();                                                                             \
        help_space->copy_to(&arg2());                                                            \
        seal();                                                                                  \
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

#define DEFINE_TWO_ARGUMENT_COMMUTATIVE_OP_FUNCTIONS(_name)                                                \
    DEFINE_TWO_ARGUMENT_OP_FUNCTIONS(_name)                                                                \
    __host__ __device__ const _name* _name::last_in_tree() const {                                         \
        if (arg1().is(Type::_name)) {                                                                      \
            return arg1().as<_name>().last_in_tree();                                                      \
        }                                                                                                  \
                                                                                                           \
        return this;                                                                                       \
    }                                                                                                      \
                                                                                                           \
    __host__ __device__ _name* _name::last_in_tree() {                                                     \
        return const_cast<_name*>(const_cast<const _name*>(this)->last_in_tree());                         \
    }                                                                                                      \
                                                                                                           \
    __host__ __device__ void _name::simplify_structure(Symbol* const help_space) {                         \
        /* TODO: Potrzebne sortowanie, inaczej nie będą poprawnie działać porównania drzew */         \
        if (!arg2().is(Type::_name)) {                                                                     \
            return;                                                                                        \
        }                                                                                                  \
                                                                                                           \
        if (!arg1().is(Type::_name)) {                                                                     \
            swap_args(help_space);                                                                         \
            return;                                                                                        \
        }                                                                                                  \
                                                                                                           \
        Symbol* const last_left = &arg1().as<_name>().last_in_tree()->arg1();                              \
                                                                                                           \
        Symbol* const last_left_copy = help_space;                                                         \
        last_left->copy_to(last_left_copy);                                                                \
        last_left->expander_placeholder = ExpanderPlaceholder::with_size(arg2().size());                   \
                                                                                                           \
        Symbol* const right_copy = help_space + last_left_copy->size();                                    \
        arg2().copy_to(right_copy);                                                                        \
        arg2().expander_placeholder = ExpanderPlaceholder::with_size(last_left_copy->size());              \
                                                                                                           \
        Symbol* const resized_reversed_this = right_copy + right_copy->size();                             \
        compress_reverse_to(resized_reversed_this);                                                        \
                                                                                                           \
        /* Zmiany w strukturze nie zmieniają całkowitego rozmiaru `this` */                              \
        Symbol::copy_and_reverse_symbol_sequence(this_symbol(), resized_reversed_this, size);              \
        right_copy->copy_to(last_left);                                                                    \
        last_left_copy->copy_to(&arg2());                                                                  \
    }                                                                                                      \
                                                                                                           \
    __host__ __device__ void _name::simplify_pairs() {                                                     \
        TreeIterator<_name, Type::_name> first(this);                                                      \
        while (first.is_valid()) {                                                                         \
            TreeIterator<_name, Type::_name> second = first;                                               \
            second.advance();                                                                              \
                                                                                                           \
            while (second.is_valid()) {                                                                    \
                if (try_fuse_symbols(first.current(), second.current())) {                                 \
                    /* Jeśli udało się coś połączyć, to upraszczanie trzeba rozpocząć od nowa     \
                     * (możnaby tylko dla zmienionego elementu, jest to opytmalizacja TODO), bo           \
                     * być może tę sumę można połączyć z czymś, co było już rozważane. Dzięki \
                     * rekurencji ogonkowej call stack nie będzie rosnąć.                               \
                     */                                                                                    \
                    return simplify_pairs();                                                               \
                }                                                                                          \
                                                                                                           \
                second.advance();                                                                          \
            }                                                                                              \
                                                                                                           \
            first.advance();                                                                               \
        }                                                                                                  \
    }

#endif
