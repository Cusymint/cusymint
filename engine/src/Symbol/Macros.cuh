#ifndef SYMBOL_DEFS_CUH
#define SYMBOL_DEFS_CUH

#include <cstring>

#include <fmt/core.h>
#include <string>
#include <type_traits>

#include "SymbolType.cuh"
#include "SimplificationResult.cuh"
#include "Utils/Cuda.cuh"
#include "Utils/Order.cuh"
#include "Utils/StaticStack.cuh"

namespace Sym {
    // Could be 0, but ininitialized memory is often equal to 0, so some checks (like in `at` in
    // `Symbol`) could fail. This way it is easier to catch problems related to unitialized memory.
    static constexpr size_t BUILDER_SIZE = std::numeric_limits<size_t>::max();

    union Symbol;
}

#define COMPRESS_REVERSE_TO_HEADER(_fname) \
    __host__ __device__ size_t _fname(Symbol* const destination) const

#define ARE_EQUAL_HEADER(_fname) __host__ __device__ bool _fname(const Symbol* const symbol) const

#define COMPARE_TO_HEADER(_fname)                                 \
    __host__ __device__ Util::Order _fname(                       \
        const Symbol& other) /* NOLINT(misc-unused-parameters) */ \
        const

#define SIMPLIFY_IN_PLACE_HEADER(_fname) __host__ __device__ bool _fname(Symbol* const help_space)

#define IS_FUNCTION_OF_HEADER(_fname)                                       \
    __host__ __device__ bool _fname(const Symbol* const* const expressions, \
                                    const size_t expression_count) const

#define PUSH_CHILDREN_ONTO_STACK_HEADER(_fname, _const) \
    __host__ __device__ void _fname(Util::StaticStack<_const Symbol*>& stack) _const

#define PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE_HEADER(_fname) \
    __host__ __device__ void _fname(Util::StaticStack<Symbol*>& stack)

#define DEFINE_PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE(_name) \
    PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE_HEADER(           \
        _name::put_children_on_stack_and_propagate_additional_size)

#define SEAL_WHOLE_HEADER(_fname) __host__ __device__ void _fname()

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
        ARE_EQUAL_HEADER(are_equal);                                              \
        COMPARE_TO_HEADER(compare_to);                                            \
        COMPRESS_REVERSE_TO_HEADER(compress_reverse_to);                          \
        SIMPLIFY_IN_PLACE_HEADER(simplify_in_place);                              \
        IS_FUNCTION_OF_HEADER(is_function_of);                                    \
        PUSH_CHILDREN_ONTO_STACK_HEADER(push_children_onto_stack, );              \
        PUSH_CHILDREN_ONTO_STACK_HEADER(push_children_onto_stack, const);         \
        PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE_HEADER(                        \
            put_children_on_stack_and_propagate_additional_size);                 \
        SEAL_WHOLE_HEADER(seal_whole);

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

#define DEFINE_ARE_EQUAL(_name) ARE_EQUAL_HEADER(_name::are_equal)

#define DEFINE_PUSH_CHILDREN_ONTO_STACK(_name)                           \
    PUSH_CHILDREN_ONTO_STACK_HEADER(_name::push_children_onto_stack, ) { \
        const_cast<const _name*>(this)->push_children_onto_stack(        \
            reinterpret_cast<Util::StaticStack<const Symbol*>&>(stack)); \
    }                                                                    \
    PUSH_CHILDREN_ONTO_STACK_HEADER(_name::push_children_onto_stack, const)

#define DEFINE_NO_OP_PUSH_CHILDREN_ONTO_STACK(_name) \
    DEFINE_PUSH_CHILDREN_ONTO_STACK(_name) {}

#define DEFINE_PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE(_name) \
    PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE_HEADER(           \
        _name::put_children_on_stack_and_propagate_additional_size)

#define DEFINE_NO_OP_PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE(_name) \
    DEFINE_PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE(_name) {}

#define DEFINE_NO_OP_SIMPLIFY_IN_PLACE(_name) \
    SIMPLIFY_IN_PLACE_HEADER(_name::simplify_in_place) { return true; } // NOLINT

#define DEFINE_SIMPLIFY_IN_PLACE(_name) SIMPLIFY_IN_PLACE_HEADER(_name::simplify_in_place)

#define DEFINE_IS_FUNCTION_OF(_name) IS_FUNCTION_OF_HEADER(_name::is_function_of)

#define DEFINE_INVALID_IS_FUNCTION_OF(_name)                                            \
    IS_FUNCTION_OF_HEADER(_name::is_function_of) /* NOLINT(misc-unused-parameters) */ { \
        Util::crash("is_function_of called on %s, this should not happen!", #_name);    \
        return false; /* Just to silence warnings */                                    \
    }

#define DEFINE_SIMPLE_ONE_ARGUMENT_IS_FUNCTION_OF(_name)                     \
    DEFINE_IS_FUNCTION_OF(_name) {                                           \
        for (size_t i = 0; i < expression_count; ++i) {                      \
            if (expressions[i]->is<_name>() &&                               \
                Symbol::are_expressions_equal(*symbol(), *expressions[i])) { \
                return true;                                                 \
            }                                                                \
        }                                                                    \
                                                                             \
        return arg().is_function_of(expressions, expression_count);          \
    }

#define BASE_ARE_EQUAL(_name) \
    symbol->type() == type && symbol->size() == size && symbol->simplified() == simplified

#define ONE_ARGUMENT_OP_ARE_EQUAL(_name) true

#define TWO_ARGUMENT_OP_ARE_EQUAL(_name) symbol->as<_name>().second_arg_offset == second_arg_offset

#define DEFINE_SIMPLE_ARE_EQUAL(_name) \
    DEFINE_ARE_EQUAL(_name) { return BASE_ARE_EQUAL(_name); }

#define DEFINE_SIMPLE_ONE_ARGUMENT_OP_ARE_EQUAL(_name) \
    DEFINE_ARE_EQUAL(_name) { return BASE_ARE_EQUAL(_name) && ONE_ARGUMENT_OP_ARE_EQUAL(_name); }

#define DEFINE_SIMPLE_TWO_ARGUMENT_OP_ARE_EQUAL(_name) \
    DEFINE_ARE_EQUAL(_name) { return BASE_ARE_EQUAL(_name) && TWO_ARGUMENT_OP_ARE_EQUAL(_name); }

#define DEFINE_COMPARE_TO(_name) COMPARE_TO_HEADER(_name::compare_to)

#define DEFINE_IDENTICAL_COMPARE_TO(_name) \
    COMPARE_TO_HEADER(_name::compare_to) { return Util::Order::Equal; }

#define DEFINE_INVALID_COMPARE_TO(_name)                                         \
    COMPARE_TO_HEADER(_name::compare_to) {                                       \
        Util::crash("compare_to called on %s, this should not happen!", #_name); \
        return Util::Order::Equal; /* Just to silence warnings */                \
    }

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

#define DEFINE_SEAL_WHOLE(_name) SEAL_WHOLE_HEADER(_name::seal_whole)

#define DEFINE_SIMPLE_SEAL_WHOLE(_name) \
    DEFINE_SEAL_WHOLE(_name) { size = 1; }

#define DEFINE_INVALID_SEAL_WHOLE(_name)                                                 \
    DEFINE_SEAL_WHOLE(_name) {                                                           \
        Util::crash("Trying to call seal_whole on '" #_name "', which is not defined."); \
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
    DEFINE_PUSH_CHILDREN_ONTO_STACK(_name) { stack.push(&arg()); }                               \
                                                                                                 \
    DEFINE_PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE(_name) {                                   \
        push_children_onto_stack(stack);                                                         \
        arg().additional_required_size() += additional_required_size;                            \
    }                                                                                            \
                                                                                                 \
    DEFINE_SEAL_WHOLE(_name) { seal(); }

#define TWO_ARGUMENT_OP_SYMBOL                                                                   \
    /* In most cases second_arg_offset == 1 + arg1().size(), but not always.                     \
     * For example in `compress_reverse_to` the first argument may not have a correct structure, \
     * but we may need the offset to the second one.                                             \
     */                                                                                          \
    size_t second_arg_offset;                                                                    \
    __host__ __device__ const Symbol& arg1() const;                                              \
    __host__ __device__ Symbol& arg1();                                                          \
    __host__ __device__ const Symbol& arg2() const;                                              \
    __host__ __device__ Symbol& arg2();                                                          \
    __host__ __device__ void seal_arg1();                                                        \
    __host__ __device__ void swap_args(Symbol* const help_space);                                \
    __host__ __device__ static void create(const Symbol* const arg1, const Symbol* const arg2,   \
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
     * @brief Checks if an operator tree is sorted                                            \
     */                                                                                       \
    __host__ __device__ bool is_tree_sorted(Symbol& help_space);                              \
                                                                                              \
    /*                                                                                        \
     * @brief W drzewie o uproszczonej strukturze wyszukuje par upraszczalnych wyrażeń.     \
     *                                                                                        \
     * @param help_space The help space                                                       \
     *                                                                                        \
     * @return `NeedsSpace` if at least one symbol needed additional space to fuse,           \
     * `NeedsSimplification` if whole expression needs to be simplified again,                \
     * `Success` otherwise. Never returns `NoAction`.                                          \
     */                                                                                       \
    __host__ __device__ SimplificationResult simplify_pairs(Symbol* const help_space);  \
                                                                                              \
    /*                                                                                        \
     * @brief Sprawdza, czy dwa drzewa można uprościć operatorem. Jeśli tak, to to robi   \
     *                                                                                        \
     * @param expr1 Pierwszy argument operatora                                               \
     * @param expr2 Drugi argument operatora                                                  \
     * @param help_space The help space                                                       \
     *                                                                                        \
     * @return `Success` jeśli wykonano uproszczenie, `NoAction`, jeśli nie,                 \
     * `NeedsSpace`, jeśli potrzeba dodatkowego miejsca na uproszczenie.                     \
     */                                                                                       \
    __host__ __device__ static SimplificationResult try_fuse_symbols(                   \
        Symbol* const expr1, Symbol* const expr2, Symbol* const help_space);                  \
    /*                                                                                        \
     * @brief Counts symbols in simplified tree.                                              \
     *                                                                                        \
     * @return Count of symbols in the tree.                                                  \
     */                                                                                       \
    __host__ __device__ size_t tree_size();

#define DEFINE_TRY_FUSE_SYMBOLS(_name)                                      \
    __host__ __device__ SimplificationResult _name::try_fuse_symbols( \
        Symbol* const expr1, Symbol* const expr2, Symbol* const help_space)

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
    DEFINE_PUSH_CHILDREN_ONTO_STACK(_name) {                                                     \
        stack.push(&arg1());                                                                     \
        stack.push(&arg2());                                                                     \
    }                                                                                            \
                                                                                                 \
    DEFINE_PUT_CHILDREN_AND_PROPAGATE_ADDITIONAL_SIZE(_name) {                                   \
        push_children_onto_stack(stack);                                                         \
        arg2().additional_required_size() += additional_required_size;                           \
    }                                                                                            \
                                                                                                 \
    DEFINE_SEAL_WHOLE(_name) {                                                                   \
        seal_arg1();                                                                             \
        seal();                                                                                  \
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
        if (!symbol()->is(_name::TYPE) ||                                                                  \
            !arg2().is(Type::_name) && is_tree_sorted(*help_space)) {                                      \
            return;                                                                                        \
        }                                                                                                  \
                                                                                                           \
        /* We can merge the subtrees like in merge sort, as both should already be                         \
         * simplified and sorted */                                                                        \
        auto left_tree_iter = ConstTreeIterator<_name>(&arg1());                                           \
        auto right_tree_iter = ConstTreeIterator<_name>(&arg2());                                          \
        const size_t arg1_height =                                                                         \
            arg1().is(_name::TYPE) ? arg1().as<_name>().tree_size() - 1 : 0;                               \
        const size_t arg2_height =                                                                         \
            arg2().is(_name::TYPE) ? arg2().as<_name>().tree_size() - 1 : 0;                               \
        const size_t new_tree_size = 1 + arg1_height + arg2_height;                                        \
                                                                                                           \
        /* Initialize a sufficient number of tree iterators */                                             \
        for (size_t i = 0; i < new_tree_size; ++i) {                                                       \
            help_space[i].init_from(_name::builder());                                                     \
        }                                                                                                  \
                                                                                                           \
        /* Because the iterators traverse the tree starting with the leaf with the largest                 \
         * distance from the root, we have to copy them to `help_space` in the same order, so that         \
         * we do not reverse the order. Also, `size` may not be equal to actual size in here               \
         * (expression may actually be smaller), this this is just an upper bound on the memory            \
         * used. */                                                                                        \
        Symbol* const help_space_back = help_space + size;                                                 \
        Symbol* current_dst_back = help_space_back;                                                        \
        while (left_tree_iter.is_valid() && right_tree_iter.is_valid()) {                                  \
            const auto ordering = Symbol::compare_expressions(                                             \
                *left_tree_iter.current(), *right_tree_iter.current(), *help_space_back);                  \
            Symbol* current;                                                                               \
            if (ordering == Util::Order::Greater) {                                                        \
                                                                                                           \
                current = left_tree_iter.current();                                                        \
                left_tree_iter.advance();                                                                  \
            }                                                                                              \
            else {                                                                                         \
                current = right_tree_iter.current();                                                       \
                right_tree_iter.advance();                                                                 \
            }                                                                                              \
                                                                                                           \
            Symbol* const current_dst = current_dst_back - current->size();                                \
            current->copy_to(current_dst);                                                                 \
            current_dst_back = current_dst;                                                                \
        }                                                                                                  \
                                                                                                           \
        auto remaining_tree_iter = left_tree_iter.is_valid() ? left_tree_iter : right_tree_iter;           \
                                                                                                           \
        while (remaining_tree_iter.is_valid()) {                                                           \
            Symbol* const current_dst = current_dst_back - remaining_tree_iter.current()->size();          \
            remaining_tree_iter.current()->copy_to(current_dst);                                           \
            remaining_tree_iter.advance();                                                                 \
            current_dst_back = current_dst;                                                                \
        }                                                                                                  \
                                                                                                           \
        /*Now we have to make sure that the copied expressions are right after the created tree            \
         * operators, otherwise the `seal`s would fail. */                                                 \
        const size_t symbols_copied = help_space_back - current_dst_back;                                  \
        Util::move_mem(help_space + new_tree_size, current_dst_back,                                       \
                       symbols_copied * sizeof(Symbol));                                                   \
                                                                                                           \
        for (size_t i = new_tree_size; i > 0; --i) {                                                       \
            help_space[i - 1].as<_name>().seal_arg1();                                                     \
            help_space[i - 1].as<_name>().seal();                                                          \
        }                                                                                                  \
                                                                                                           \
        help_space->copy_to(symbol());                                                                     \
    }                                                                                                      \
                                                                                                           \
    __host__ __device__ bool _name::is_tree_sorted(Symbol& help_space) {                                   \
        TreeIterator<_name> iterator(this);                                                                \
        Symbol* last = iterator.current();                                                                 \
        iterator.advance();                                                                                \
                                                                                                           \
        while (iterator.is_valid()) {                                                                      \
            if (Symbol::compare_expressions(*last, *iterator.current(), help_space) ==                     \
                Util::Order::Less) {                                                                       \
                return false;                                                                              \
            }                                                                                              \
                                                                                                           \
            last = iterator.current();                                                                     \
            iterator.advance();                                                                            \
        }                                                                                                  \
                                                                                                           \
        return true;                                                                                       \
    }                                                                                                      \
                                                                                                           \
    __host__ __device__ SimplificationResult _name::simplify_pairs(                                  \
        Symbol* const help_space) {                                                                        \
        bool expression_changed = true;                                                                    \
        SimplificationResult result = SimplificationResult::Success;                           \
        while (expression_changed) {                                                                       \
            expression_changed = false;                                                                    \
            TreeIterator<_name> first(this);                                                               \
            while (first.is_valid()) {                                                                     \
                TreeIterator<_name> second = first;                                                        \
                second.advance();                                                                          \
                                                                                                           \
                while (second.is_valid()) {                                                                \
                    switch (try_fuse_symbols(first.current(), second.current(), help_space)) {             \
                    case SimplificationResult::Success:                                              \
                        /* Jeśli udało się coś połączyć, to upraszczanie trzeba rozpocząć od nowa \
                         * (możnaby tylko dla zmienionego elementu, jest to opytmalizacja TODO),          \
                         * bo być może tę sumę można połączyć z czymś, co było już rozważane.  \
                         */                                                                                \
                        expression_changed = true;                                                         \
                        break;                                                                             \
                    case SimplificationResult::NeedsSimplification:                                  \
                        result = SimplificationResult::NeedsSimplification;                          \
                        break;                                                                             \
                    case SimplificationResult::NeedsSpace:                                           \
                        result = SimplificationResult::NeedsSpace;                                   \
                        break;                                                                             \
                    case SimplificationResult::NoAction:                                              \
                        break;                                                                             \
                    }                                                                                      \
                                                                                                           \
                    second.advance();                                                                      \
                }                                                                                          \
                                                                                                           \
                first.advance();                                                                           \
            }                                                                                              \
        }                                                                                                  \
        return result;                                                                                     \
    }                                                                                                      \
                                                                                                           \
    /*                                                                                                     \
     * @brief Number of leaves in a two argument operator tree                                             \
     */                                                                                                    \
    __host__ __device__ size_t _name::tree_size() {                                                        \
        /* In every sum, number of terms is equal to number of operator signs plus 1.                      \
         * When an addition tree is simplified, all operator symbols are placed in a row,                  \
         * so it suffices to calculate address of the last operator symbol. The offset between             \
         * `this` and last plus 1 is the number of operator signs in the sum.                              \
         * Thus, the offset plus 2 is the number of terms in the sum.                                      \
         * Conversion to `Symbol*` with `symbol()` function is necessary, because                          \
         * `_name`                                                                                         \
         * structure may be smaller than `Symbol` union.                                                   \
         */                                                                                                \
        return last_in_tree()->symbol() - symbol() + 2;                                                    \
    }

#endif
