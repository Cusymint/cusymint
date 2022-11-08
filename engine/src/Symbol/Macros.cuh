#ifndef SYMBOL_DEFS_CUH
#define SYMBOL_DEFS_CUH

#include <cstring>

#include <fmt/core.h>
#include <string>
#include <type_traits>

#include "SymbolType.cuh"
#include "Utils/Cuda.cuh"
#include "Utils/StaticStack.cuh"

namespace Sym {
    // Could be 0, but ininitialized memory is often equal to 0, so some checks (like in `at` in
    // `Symbol`) could fail. This way it is easier to catch problems related to unitialized memory.
    static constexpr size_t BUILDER_SIZE = std::numeric_limits<size_t>::max();

    union Symbol;
}

#define COMPRESS_REVERSE_TO_HEADER(_compress_reverse_to) \
    __host__ __device__ size_t _compress_reverse_to(Symbol* const destination) const

#define COMPARE_HEADER(_compare) __host__ __device__ bool _compare(const Symbol* const symbol) const

#define SIMPLIFY_IN_PLACE_HEADER(_simplify_in_place) \
    __host__ __device__ bool _simplify_in_place(Symbol* const help_space)

#define IS_FUNCTION_OF_HEADER(_is_function_of)                                       \
    __host__ __device__ bool _is_function_of(const Symbol* const* const expressions, \
                                             const size_t expression_count) const

#define PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE_HEADER(_fname) \
    __host__ __device__ void _fname(Util::StaticStack<Symbol*>& stack)

#define DEFINE_PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE(_name) \
    PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE_HEADER(           \
        _name::put_children_on_stack_and_propagate_additional_size)

#define DECLARE_SYMBOL(_name, _simple)                                            \
    struct _name {                                                                \
        constexpr static Sym::Type TYPE = Sym::Type::_name;                       \
        Sym::Type type;                                                           \
        size_t size;                                                              \
        bool simplified;                                                          \
        bool to_be_copied;                                                        \
        size_t additional_required_size;                                          \
                                                                                  \
        __host__ __device__ static _name builder() {                              \
            return {                                                              \
                .type = Sym::Type::_name,                                         \
                .size = BUILDER_SIZE,                                             \
                .simplified = _simple,                                            \
                .to_be_copied = false,                                            \
                .additional_required_size = 0,                                    \
            };                                                                    \
        }                                                                         \
                                                                                  \
        __host__ __device__ void seal();                                          \
                                                                                  \
        __host__ __device__ static _name create() {                               \
            return {                                                              \
                .type = Sym::Type::_name,                                         \
                .size = 1,                                                        \
                .simplified = _simple,                                            \
                .to_be_copied = false,                                            \
                .additional_required_size = 0,                                    \
            };                                                                    \
        }                                                                         \
                                                                                  \
        __host__ __device__ inline const Symbol* symbol() const {                 \
            return reinterpret_cast<const Symbol*>(this);                         \
        }                                                                         \
                                                                                  \
        __host__ __device__ inline Symbol* symbol() {                             \
            return const_cast<Symbol*>(const_cast<const _name*>(this)->symbol()); \
        }                                                                         \
                                                                                  \
        template <class T> __host__ __device__ inline const T* as() const {       \
            return reinterpret_cast<const T*>(this);                              \
        }                                                                         \
                                                                                  \
        template <class T> __host__ __device__ inline T* as() {                   \
            return const_cast<T*>(const_cast<const _name*>(this)->as<T>());       \
        }                                                                         \
                                                                                  \
        COMPARE_HEADER(compare);                                                  \
        COMPRESS_REVERSE_TO_HEADER(compress_reverse_to);                          \
        SIMPLIFY_IN_PLACE_HEADER(simplify_in_place);                              \
        IS_FUNCTION_OF_HEADER(is_function_of);                                    \
        PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE_HEADER(                        \
            put_children_on_stack_and_propagate_additional_size);

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

#define DEFINE_NO_OP_PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE(_name) \
    DEFINE_PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE(_name) {}

#define DEFINE_NO_OP_SIMPLIFY_IN_PLACE(_name) \
    SIMPLIFY_IN_PLACE_HEADER(_name::simplify_in_place) { return true; } // NOLINT

#define DEFINE_SIMPLIFY_IN_PLACE(_name) SIMPLIFY_IN_PLACE_HEADER(_name::simplify_in_place)

#define DEFINE_IS_FUNCTION_OF(_name) IS_FUNCTION_OF_HEADER(_name::is_function_of)

#define DEFINE_INVALID_IS_FUNCTION_OF(_name)                                         \
    IS_FUNCTION_OF_HEADER(_name::is_function_of) {                                   \
        Util::crash("Is function of called on %s, this should not happen!", #_name); \
        return false; /* Just to silence warnings */                                 \
    }

#define DEFINE_SIMPLE_ONE_ARGUMENT_IS_FUNCTION_OF(_name)                       \
    DEFINE_IS_FUNCTION_OF(_name) {                                             \
        for (size_t i = 0; i < expression_count; ++i) {                        \
            if (expressions[i]->is<_name>() && *symbol() == *expressions[i]) { \
                return true;                                                   \
            }                                                                  \
        }                                                                      \
                                                                               \
        return arg().is_function_of(expressions, expression_count);            \
    }

#define BASE_COMPARE(_name) \
    symbol->type() == type && symbol->size() == size && symbol->simplified() == simplified

#define ONE_ARGUMENT_OP_COMPARE(_name) true

#define TWO_ARGUMENT_OP_COMPARE(_name) symbol->as<_name>().second_arg_offset == second_arg_offset

#define DEFINE_SIMPLE_COMPARE(_name) \
    DEFINE_COMPARE(_name) { return BASE_COMPARE(_name); }

#define DEFINE_SIMPLE_ONE_ARGUMETN_OP_COMPARE(_name) \
    DEFINE_COMPARE(_name) { return BASE_COMPARE(_name) && ONE_ARGUMENT_OP_COMPARE(_name); }

#define DEFINE_SIMPLE_TWO_ARGUMENT_OP_COMPARE(_name) \
    DEFINE_COMPARE(_name) { return BASE_COMPARE(_name) && TWO_ARGUMENT_OP_COMPARE(_name); }

#define DEFINE_TO_STRING(_str) \
    [[nodiscard]] std::string to_string() const { return _str; }

#define DEFINE_TO_TEX(_str) \
    [[nodiscard]] std::string to_tex() const { return _str; }

#define DEFINE_COMPRESS_REVERSE_TO(_name) COMPRESS_REVERSE_TO_HEADER(_name::compress_reverse_to)

#define DEFINE_SIMPLE_COMPRESS_REVERSE_TO(_name)                                \
    DEFINE_COMPRESS_REVERSE_TO(_name) {                                         \
        for (size_t i = 0; i < additional_required_size; ++i) {                 \
            (destination + i)->init_from(Unknown::create());                    \
        }                                                                       \
        Symbol* const new_destination = destination + additional_required_size; \
        symbol()->copy_single_to(new_destination);                              \
        return size + additional_required_size;                                 \
    }

#define DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(_name)                                  \
    DEFINE_COMPRESS_REVERSE_TO(_name) {                                                    \
        size_t& child_additional_size = (destination - 1)->additional_required_size();     \
        if (child_additional_size > additional_required_size) {                            \
            (destination - 1)->size() += child_additional_size - additional_required_size; \
        }                                                                                  \
        child_additional_size = 0;                                                         \
                                                                                           \
        const size_t new_arg_size = (destination - 1)->size();                             \
        symbol()->copy_single_to(destination);                                             \
        destination->size() = new_arg_size + 1;                                            \
        return 1;                                                                          \
    }

#define DEFINE_TWO_ARGUMENT_OP_COMPRESS_REVERSE_TO(_name)                           \
    DEFINE_COMPRESS_REVERSE_TO(_name) {                                             \
        (destination - 1)->size() += (destination - 1)->additional_required_size(); \
        (destination - 1)->additional_required_size() = 0;                          \
                                                                                    \
        const size_t new_arg1_size = (destination - 1)->size();                     \
        size_t& child_additional_size =                                             \
            (destination - new_arg1_size - 1)->additional_required_size();          \
        if (child_additional_size > additional_required_size) {                     \
            (destination - new_arg1_size - 1)->size() +=                            \
                child_additional_size - additional_required_size;                   \
        }                                                                           \
        child_additional_size = 0;                                                  \
                                                                                    \
        const size_t new_arg2_size = (destination - new_arg1_size - 1)->size();     \
                                                                                    \
        symbol()->copy_single_to(destination);                                      \
        destination->size() = new_arg1_size + new_arg2_size + 1;                    \
        destination->as<_name>().second_arg_offset = new_arg1_size + 1;             \
        return 1;                                                                   \
    }

#define DEFINE_UNSUPPORTED_COMPRESS_REVERSE_TO(_name)                                     \
    DEFINE_COMPRESS_REVERSE_TO(_name) {                                                   \
        Util::crash("ERROR: compress_reverse_to used on unsupported type: %s\n", #_name); \
        return 0;                                                                         \
    }

#define DEFINE_INTO_DESTINATION_OPERATOR(_name)                                        \
    __host__ __device__ _name* operator<<(Symbol* const destination, _name&& target) { \
        destination->init_from(target);                                                \
        return &destination->as<_name>();                                              \
    }                                                                                  \
                                                                                       \
    __host__ __device__ _name* operator<<(Symbol& destination, _name&& target) {       \
        destination.init_from(target);                                                 \
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
    }                                                                                            \
                                                                                                 \
    DEFINE_PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE(_name) {                                   \
        stack.push(&arg());                                                                      \
        arg().additional_required_size() += additional_required_size;                            \
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

#define TWO_ARGUMENT_COMMUTATIVE_OP_SYMBOL(_name)                                               \
    TWO_ARGUMENT_OP_SYMBOL                                                                      \
    /*                                                                                          \
     * @brief W uproszczonym drzewie operatora zwraca operację najniżej w drzewie             \
     *                                                                                          \
     * @return Wskaźnik do ostatniego operator. Jeśli `arg1()` nie jest tego samego typu co   \
     * `*this`, to zwraca `this`                                                                \
     */                                                                                         \
    __host__ __device__ const _name* last_in_tree() const;                                      \
                                                                                                \
    /*                                                                                          \
     * @brief Przeładowanie bez `const`                                                        \
     */                                                                                         \
    __host__ __device__ _name* last_in_tree();                                                  \
                                                                                                \
    /*                                                                                          \
     * @brief Uproszczenie struktury operatora `$` do drzewa w postaci:                         \
     *                 $                                                                        \
     *                / \                                                                       \
     *               $   e                                                                      \
     *              / \                                                                         \
     *             $   d                                                                        \
     *            / \                                                                           \
     *           $   c                                                                          \
     *          / \                                                                             \
     *         a   b                                                                            \
     *                                                                                          \
     * Zakładamy, że oba argumenty są w uproszczonej postaci                                 \
     *                                                                                          \
     * @param help_space Pamięć pomocnicza                                                    \
     */                                                                                         \
    __host__ __device__ void simplify_structure(Symbol* const help_space);                      \
                                                                                                \
    /*                                                                                          \
     * @brief W drzewie o uproszczonej strukturze wyszukuje par upraszczalnych wyrażeń.       \
     */                                                                                         \
    __host__ __device__ void simplify_pairs();                                                  \
                                                                                                \
    /*                                                                                          \
     * @brief Sprawdza, czy dwa drzewa można uprościć operatorem. Jeśli tak, to to robi     \
     *                                                                                          \
     * @param expr1 Pierwszy argument operatora                                                 \
     * @param expr2 Drugi argument operatora                                                    \
     *                                                                                          \
     * @return `true` jeśli wykonano uproszczenie, `false` w przeciwnym wypadku                \
     */                                                                                         \
    __host__ __device__ static bool try_fuse_symbols(Symbol* const expr1, Symbol* const expr2); \
    /*                                                                                          \
     * @brief Counts symbols in simplified tree.                                                \
     *                                                                                          \
     * @return Count of symbols in the tree.                                                    \
     */                                                                                         \
    __host__ __device__ size_t tree_size();

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
    }                                                                                            \
                                                                                                 \
    DEFINE_PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE(_name) {                                   \
        stack.push(&arg1());                                                                     \
        stack.push(&arg2());                                                                     \
        arg2().additional_required_size() += additional_required_size;                           \
    }

#define DEFINE_TWO_ARGUMENT_COMMUTATIVE_OP_FUNCTIONS(_name)                                                \
    DEFINE_TWO_ARGUMENT_OP_FUNCTIONS(_name)                                                                \
    __host__ __device__ const _name* _name::last_in_tree() const {                                         \
        auto* last = this;                                                                                 \
        while (last->arg1().is(Type::_name)) {                                                             \
            last = last->arg1().as_ptr<_name>();                                                           \
        }                                                                                                  \
                                                                                                           \
        return last;                                                                                       \
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
        const size_t new_size = symbol()->compress_reverse_to(resized_reversed_this);                      \
                                                                                                           \
        /* Zmiany w strukturze mogą zmienić całkowity rozmiar `this`, bo                                \
         * compress_reverse_to skróci wcześniej uproszczone wyrażenia*/                                 \
        Symbol::copy_and_reverse_symbol_sequence(symbol(), resized_reversed_this, new_size);               \
        right_copy->copy_to(last_left);                                                                    \
        last_left_copy->copy_to(&arg2());                                                                  \
    }                                                                                                      \
                                                                                                           \
    __host__ __device__ void _name::simplify_pairs() {                                                     \
        bool expression_changed = true;                                                                    \
        while (expression_changed) {                                                                       \
            expression_changed = false;                                                                    \
            TreeIterator<_name> first(this);                                                               \
            while (first.is_valid()) {                                                                     \
                TreeIterator<_name> second = first;                                                        \
                second.advance();                                                                          \
                                                                                                           \
                while (second.is_valid()) {                                                                \
                    if (try_fuse_symbols(first.current(), second.current())) {                             \
                        /* Jeśli udało się coś połączyć, to upraszczanie trzeba rozpocząć od nowa \
                         * (możnaby tylko dla zmienionego elementu, jest to opytmalizacja TODO),          \
                         * bo być może tę sumę można połączyć z czymś, co było już rozważane.  \
                         * Dzięki rekurencji ogonkowej call stack nie będzie rosnąć.                   \
                         */                                                                                \
                        /*return simplify_pairs();*/                                                       \
                        expression_changed = true;                                                         \
                    }                                                                                      \
                                                                                                           \
                    second.advance();                                                                      \
                }                                                                                          \
                                                                                                           \
                first.advance();                                                                           \
            }                                                                                              \
        }                                                                                                  \
    }                                                                                                      \
                                                                                                           \
    __host__ __device__ size_t _name::tree_size() {                                                        \
        return last_in_tree()->symbol() - symbol() + 2;                                                    \
    }

#endif
