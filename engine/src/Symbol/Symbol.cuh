#ifndef SYMBOL_CUH
#define SYMBOL_CUH

#include <functional>
#include <string>
#include <vector>

#include "Addition.cuh"
#include "Constants.cuh"
#include "ExpanderPlaceholder.cuh"
#include "Hyperbolic.cuh"
#include "Integral.cuh"
#include "InverseTrigonometric.cuh"
#include "Logarithm.cuh"
#include "Polynomial.cuh"
#include "Power.cuh"
#include "Product.cuh"
#include "Sign.cuh"
#include "Solution.cuh"
#include "SubexpressionCandidate.cuh"
#include "SubexpressionVacancy.cuh"
#include "Substitution.cuh"
#include "SymbolType.cuh"
#include "Trigonometric.cuh"
#include "Unknown.cuh"
#include "Variable.cuh"

#include "Utils/CompileConstants.cuh"
#include "Utils/Cuda.cuh"
#include "Utils/OptionalNumber.cuh"
#include "Utils/Result.cuh"

#define VC_CASE(_type, _instance, _member_function, ...)              \
    case Type::_type: {                                               \
        return (_instance).as<_type>()._member_function(__VA_ARGS__); \
    }

// Also works when `_member_function` returns void
#define VIRTUAL_CALL(_instance, _member_function, ...)                                     \
    (([&]() {                                                                              \
        switch ((_instance).type()) {                                                      \
        case Type::Symbol: {                                                               \
            Util::crash("Trying to access a virtual function (%s) on a pure Symbol",       \
                        #_member_function);                                                \
            break;                                                                         \
        }                                                                                  \
            VC_CASE(NumericConstant, _instance, _member_function, __VA_ARGS__)             \
            VC_CASE(KnownConstant, _instance, _member_function, __VA_ARGS__)               \
            VC_CASE(UnknownConstant, _instance, _member_function, __VA_ARGS__)             \
            VC_CASE(Variable, _instance, _member_function, __VA_ARGS__)                    \
            VC_CASE(ExpanderPlaceholder, _instance, _member_function, __VA_ARGS__)         \
            VC_CASE(SubexpressionCandidate, _instance, _member_function, __VA_ARGS__)      \
            VC_CASE(SubexpressionVacancy, _instance, _member_function, __VA_ARGS__)        \
            VC_CASE(Integral, _instance, _member_function, __VA_ARGS__)                    \
            VC_CASE(Solution, _instance, _member_function, __VA_ARGS__)                    \
            VC_CASE(Substitution, _instance, _member_function, __VA_ARGS__)                \
            VC_CASE(Addition, _instance, _member_function, __VA_ARGS__)                    \
            VC_CASE(Negation, _instance, _member_function, __VA_ARGS__)                    \
            VC_CASE(Product, _instance, _member_function, __VA_ARGS__)                     \
            VC_CASE(Reciprocal, _instance, _member_function, __VA_ARGS__)                  \
            VC_CASE(Power, _instance, _member_function, __VA_ARGS__)                       \
            VC_CASE(Sign, _instance, _member_function, __VA_ARGS__)                        \
            VC_CASE(Sine, _instance, _member_function, __VA_ARGS__)                        \
            VC_CASE(Cosine, _instance, _member_function, __VA_ARGS__)                      \
            VC_CASE(Tangent, _instance, _member_function, __VA_ARGS__)                     \
            VC_CASE(Cotangent, _instance, _member_function, __VA_ARGS__)                   \
            VC_CASE(Arcsine, _instance, _member_function, __VA_ARGS__)                     \
            VC_CASE(Arccosine, _instance, _member_function, __VA_ARGS__)                   \
            VC_CASE(Arctangent, _instance, _member_function, __VA_ARGS__)                  \
            VC_CASE(Arccotangent, _instance, _member_function, __VA_ARGS__)                \
            VC_CASE(Logarithm, _instance, _member_function, __VA_ARGS__)                   \
            VC_CASE(Polynomial, _instance, _member_function, __VA_ARGS__)                  \
            VC_CASE(Unknown, _instance, _member_function, __VA_ARGS__)                     \
        }                                                                                  \
                                                                                           \
        Util::crash("Trying to access a virtual function (%s) on an invalid type",         \
                    #_member_function);                                                    \
        /* To avoid warnings about missing return, it is not going to be called anyways */ \
        return (_instance).unknown._member_function(__VA_ARGS__);                          \
    })())

namespace Sym {
    union Symbol {
        Unknown unknown;
        NumericConstant numeric_constant;
        KnownConstant known_constant;
        UnknownConstant unknown_constant;
        Variable variable;
        ExpanderPlaceholder expander_placeholder;
        Integral integral;
        Solution solution;
        Substitution substitution;
        SubexpressionCandidate subexpression_candidate;
        SubexpressionVacancy subexpression_vacancy;
        Addition addition;
        Negation negation;
        Product product;
        Reciprocal reciprocal;
        Power power;
        Sign sign;
        Sine sine;
        Cosine cosine;
        Tangent tangent;
        Cotangent cotangent;
        Arcsine arcsine;
        Arccosine arccosine;
        Arctangent arctangent;
        Arccotangent arccotangent;
        Logarithm logarithm;
        Polynomial polynomial;

        constexpr static Sym::Type TYPE = Sym::Type::Symbol;

        [[nodiscard]] __host__ __device__ inline Type type() const { return unknown.type; }
        [[nodiscard]] __host__ __device__ inline size_t type_ordinal() const {
            return Sym::type_ordinal(type());
        }

        [[nodiscard]] __host__ __device__ inline bool simplified() const {
            return unknown.simplified;
        }

        [[nodiscard]] __host__ __device__ inline bool is(const Type other_type) const {
            return type() == other_type;
        }

        [[nodiscard]] __host__ __device__ bool is(const double number) const;

        [[nodiscard]] __host__ __device__ bool is_integer() const;

        template <class T> [[nodiscard]] __host__ __device__ inline bool is() const {
            return type() == T::TYPE;
        }

        [[nodiscard]] __host__ __device__ inline size_t& size() { return unknown.size; }

        [[nodiscard]] __host__ __device__ inline size_t size() const { return unknown.size; }

        [[nodiscard]] __host__ __device__ inline size_t& additional_required_size() {
            return unknown.additional_required_size;
        }

        [[nodiscard]] __host__ __device__ inline size_t additional_required_size() const {
            return unknown.additional_required_size;
        }

        [[nodiscard]] __host__ __device__ inline bool& to_be_copied() {
            return unknown.to_be_copied;
        }

        [[nodiscard]] __host__ __device__ inline bool to_be_copied() const {
            return unknown.to_be_copied;
        }

        template <class T> __host__ __device__ inline T& init_from(const T& other) {
            // Not using `as<>` to prevent errors, as in this case
            // `this` can be of a type different than T
            *reinterpret_cast<T*>(this) = other;
            return *reinterpret_cast<T*>(this);
        };

        template <class T>
        [[nodiscard]] __host__ __device__ static inline const Symbol* from(const T* sym) {
            return reinterpret_cast<const Symbol*>(sym);
        }

        template <class T> [[nodiscard]] __host__ __device__ static inline Symbol* from(T* sym) {
            return const_cast<Symbol*>(from(const_cast<const T*>(sym)));
        }

        template <class T> [[nodiscard]] __host__ __device__ inline const T* as_ptr() const {
            return reinterpret_cast<const T*>(this);
        }

        template <class T> [[nodiscard]] __host__ __device__ inline T* as_ptr() {
            return const_cast<T*>(const_cast<const Symbol*>(this)->as_ptr<T>());
        }

        template <class T> [[nodiscard]] __host__ __device__ inline const T& as() const {
            if constexpr (Consts::DEBUG) {
                if (T::TYPE != Type::Symbol && T::TYPE != type()) {
                    Util::crash("Trying to access %s as %s", type_name(type()), type_name(T::TYPE));
                }
            }

            return *as_ptr<T>();
        }

        template <class T> [[nodiscard]] __host__ __device__ inline T& as() {
            return const_cast<T&>(const_cast<const Symbol&>(*this).as<T>());
        }

        /*
         * @brief Pointer to the `idx`th element after `this`, without bounds checking in debug
         */
        [[nodiscard]] __host__ __device__ inline const Symbol*
        at_unchecked(const size_t idx) const {
            return this + idx;
        }

        /*
         * @brief Pointer to the `idx`th element after `this`, without bounds checking in debug
         */
        [[nodiscard]] __host__ __device__ inline Symbol* at_unchecked(const size_t idx) {
            return const_cast<Symbol*>(const_cast<const Symbol*>(this)->at_unchecked(idx));
        }

        /*
         * @brief Pointer to the `idx`th element after `this`.
         */
        [[nodiscard]] __host__ __device__ inline const Symbol* at(const size_t idx) const {
            if constexpr (Consts::DEBUG) {
                // If `this` is under construction, we allow access to symbols after it without
                // checks
                if (size() != BUILDER_SIZE && size() <= idx) {
                    Util::crash(
                        "Trying to access element at index %lu after a symbol, but the symbol's "
                        "size is %lu and it is not under construction",
                        idx, size());
                }
            }

            return at_unchecked(idx);
        }

        /*
         * @brief Pointer to the `idx`th element after `this`.
         */
        [[nodiscard]] __host__ __device__ inline Symbol* at(const size_t idx) {
            return const_cast<Symbol*>(const_cast<const Symbol*>(this)->at(idx));
        }

        /*
         * @brief Reference to the `idx`th element after `this`
         */
        [[nodiscard]] __host__ __device__ inline const Symbol& operator[](const size_t idx) const {
            return *at(idx);
        }

        /*
         * @brief Reference to the `idx`th element after `this`
         */
        [[nodiscard]] __host__ __device__ inline Symbol& operator[](const size_t idx) {
            return const_cast<Symbol&>(const_cast<const Symbol*>(this)->operator[](idx));
        }

        /*
         * @brief Pointer to the symbol right after `this`
         */
        [[nodiscard]] __host__ __device__ inline const Symbol& child() const {
            return this->operator[](1);
        }

        /*
         * @brief Pointer to the symbol right after `this`
         */
        [[nodiscard]] __host__ __device__ inline Symbol& child() {
            return const_cast<Symbol&>(const_cast<const Symbol*>(this)->child());
        }

        /*
         * @brief Copies symbol sequence from `source` to `destination`.Source and destination
         * cannot alias.
         *
         * @param seq Symbol sequence to copy. Doesn't need to be semantically correct.
         * @param n number of symbols to copy
         */
        __host__ __device__ static void copy_symbol_sequence(Symbol* const destination,
                                                             const Symbol* const source, size_t n);

        /*
         * @brief Moves symbol sequence from `source` to `destination`. Source and destination
         * can alias.
         *
         * @param seq Symbol sequence to copy. Doesn't need to be semantically correct.
         * @param n number of symbols to copy
         */
        __host__ __device__ static void
        move_symbol_sequence(Symbol* const destination, Symbol* const source, size_t symbol_count);

        /*
         * @brief Copies symbol sequence from `source` to `destination` in reverse.
         *
         * @param seq Symbol sequence to copy. Doesn't need to be semantically correct.
         * @param n number of symbols to copy
         */
        __host__ __device__ static void
        copy_and_reverse_symbol_sequence(Symbol& destination, const Symbol& source, const size_t n);

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
        __host__ __device__ static bool compare_symbol_sequences(const Symbol* seq1,
                                                                 const Symbol* seq2, size_t n);

        /*
         * @brief Swap contents of two symbols
         *
         * @param symbol1 First symbol
         * @param symbol2 Second symbol
         */
        __host__ __device__ static void swap_symbols(Symbol* const symbol1, Symbol* const symbol2);

        /*
         * @brief Reverses a symbol sequence in place.
         *
         * @param seq Symbol sequence to reverse. Doesn't need to be semantically correct.
         * @param n Sequence length
         */
        __host__ __device__ static void reverse_symbol_sequence(Symbol* const seq, size_t n);

        /*
         * @brief Copies `*this` into `destination`. Does not copy the whole tree, only a single
         * symbol.
         *
         * @param destination Copy destination
         */
        __host__ __device__ void copy_single_to(Symbol& destination) const;

        /*
         * @brief Copies `this` into `destination`. Copies the whole tree.
         *
         * @param destination Copy destination. Cannot alias with `this`.
         */
        __host__ __device__ void copy_to(Symbol& destination) const;

        /*
         * @brief Copies `this` into `destination`. Copies the whole tree.
         *
         * @param destination Copy destination. Can alias with `this`.
         */
        __host__ __device__ void move_to(Symbol& destination);

        /*
         * @brief Checks if `this` is an expression composed only of constants
         *
         * @return `true` if `this` has no variables, `false` otherwise
         */
        [[nodiscard]] __host__ __device__ bool is_constant() const;

        /*
         * @brief Returns offset of first occurence of variable in this symbol sequence
         *
         * @return Offset of first variable symbol. If there is none, then -1.
         */
        [[nodiscard]] __host__ __device__ ssize_t first_var_occurence() const;

        /*
         * @brief Checks if `this` expression is a function of expressions in `expressions`.
         * More formally, if `this` is expressed as `f(x)` and expressions in `expressions` are
         * expressed as `g1(x), g2(x), ..., gn(x)`, this function checks if there exists a
         * function `h(x1, x2, ..., xn)`, such that `f(x) = h(g1(x), g2(x), ..., gn(x))`.
         *
         * @param expressions Expressions to check
         *
         * @return `true` if `this` is a function of `expressions`, false otherwise. Returns
         * false also when `this` is constant. In particular, it returns true if `this` is a
         * constant expression
         */
        template <class... Args, std::enable_if_t<(std::is_same_v<Args, Symbol> && ...), int> = 0>
        [[nodiscard]] __host__ __device__ bool is_function_of(const Args&... expressions) const {
            const Symbol* const expression_array[] = {&expressions...};
            return VIRTUAL_CALL(*this, is_function_of, expression_array, sizeof...(Args));
        }

        /*
         * @brief Another version of `is_function_of` that is the same as the functions defined
         * in concrete symbols. It should be called only when the templated version cannot be
         * used.
         *
         * @param expressions Array of pointers to expressions to check
         * @param expression_count Number of expressions in `expressions`
         *
         * @return Same as in the other function
         */
        [[nodiscard]] __host__ __device__ bool
        is_function_of(const Symbol* const* const expressions, const size_t expression_count) const;

        /*
         * @brief Removes holes from symbol tree and copies it in reverse order to
         * `destination`.
         *
         * @param destination Location to which the tree is going to be copied
         *
         * @return New size of the symbol tree when the comppression was successfull, an error
         * when the destination's capacity is too small
         */
        [[nodiscard]] __host__ __device__ Util::SimpleResult<size_t>
        compress_reverse_to(SymbolIterator destination);

        /*
         * @brief Removes holes from symbol tree and copies it to `destination`.
         *
         * @param destination Location to which the tree is going to be copied. Cannot be same
         * as `this`.
         *
         * @return New size of the symbol tree
         */
        [[nodiscard]] __host__ __device__ Util::SimpleResult<size_t>
        compress_to(SymbolIterator& destination);

        __host__ __device__ void
        mark_to_be_copied_and_propagate_additional_size(Symbol* const help_space);

        /*
         * @brief Simplifies an expression
         *
         * @param help_space Help space
         * @param
         *
         * @return A good result when the simplification succeds, an error result when the
         * help_space is too small or when the simplification result would be larger than
         * `capacity`
         */
        [[nodiscard]] __host__ __device__ Util::BinaryResult simplify(SymbolIterator& help_space);

        /*
         * @brief Simplified an expression. Can leave the expression with holes.
         *
         * @param help_space Help space
         *
         * @return `true` if expression was simplified, `false` if simplified result
         * would take more space than `size()` or expression needs to be simplified again.
         */
        [[nodiscard]] __host__ __device__ bool simplify_in_place(SymbolIterator& help_space);

        /*
         * @brief Recalculates sizes and argument offsets in the given expression. There cannot
         * be any holes in the expression.
         *
         * @param expr Expression to seal
         * @param size Total size of the expression
         */
        __host__ __device__ static void seal_whole(Symbol& expr, const size_t size);

        /*
         * @brief Substitutes all occurences of variable with `symbol`
         *
         * @param symbol Symbol to substitute variables with, cannot have any children
         */
        void substitute_variable_with(const Symbol symbol);

        /*
         * @brief A wrapper for `substitute_variable_with` faciliating creation of strings with
         * substitutions.
         *
         * @param n Number of substitution to take the name from.
         */
        void substitute_variable_with_nth_substitution_name(const size_t n);

        /*
         * @brief Jeśli ten symbol jest typu T, to wykonaj na nim `function`
         *
         * @param function Funkcja która się wykona jeśli `this` jest typu T. Jako argument
         * podawane jest `this`
         */
        template <class T, class F> __host__ __device__ void if_is_do(F function) {
            if (T::TYPE == type()) {
                function(as<T>());
            }
        }

        /*
         * @brief Jeśli ten symbol jest typu T, to wykonaj na nim `function`
         *
         * @param function Funkcja która się wykona jeśli `this` jest typu T. Jako argument
         * podawane jest `this`
         */
        template <class T, class F> __host__ __device__ void if_is_do(F function) const {
            if (T::TYPE == type()) {
                function(as<T>());
            }
        }

        /*
         * @brief Checks if two expressions are equal
         *
         * @param expr1 First expression
         * @param expr2 Second expression
         *
         * @return `true` if expressions have the same structure, `false` otherwise
         */
        [[nodiscard]] __host__ __device__ static bool are_expressions_equal(const Symbol& expr1,
                                                                            const Symbol& expr2);

        /*
         * @brief Compares two expressions. Size fields in expressions do not need to be valid,
         * but have to indicate a size that is not smaller than the real size.
         *
         * @param expr1 First expression
         * @param expr2 Second expression
         * @param help_space Help space used for comparing (necessary because expressions are
         * allowed to have invalid sizes)
         *
         * @return `Util::Order::Less` if `expr1` comes before `expr2` in the expression
         * order, `Util::Order::Greater` if `expr2` comes before `expr1`, and
         * `Util::Order::Equal`, if expressions are equal
         */
        [[nodiscard]] __host__ __device__ static Util::Order
        compare_expressions(const Symbol& expr1, const Symbol& expr2, Symbol& help_space);

        /*
         * @brief String formatting of the expression
         */
        [[nodiscard]] std::string to_string() const;

        /*
         * @brief TeX string formatting of the expression
         */
        [[nodiscard]] std::string to_tex() const;

        /*
         * @brief Checks if `this` is a polynomial. Returns its rank if yes. Otherwise, returns
         * `empty_num`.
         */
        __host__ __device__ Util::OptionalNumber<ssize_t>
        is_polynomial(Symbol* const help_space) const;

        /*
         * @brief If `this` is a monomial, returns its coefficient. Otherwise, returns
         * `empty_num`.
         */
        __host__ __device__ Util::OptionalNumber<double>
        get_monomial_coefficient(Symbol* const help_space) const;

        /*
         * @brief Calculates derivative of `this` and places it at `destination`.
         *
         * @param `destination` This is what it is.
         *
         * @return Number of symbols inserted.
         */
        __host__ __device__ size_t derivative_to(Symbol* const destination);
    };

    /*
     * @brief Compares two symbols (does not compare expressions!)
     *
     * @param sym1 First symbol
     * @param sym2 Second symbol
     *
     * @return `true` if symbols are equal, `false` otherwise
     */
    __host__ __device__ bool operator==(const Symbol& sym1, const Symbol& sym2);

    /*
     * @brief Compares two symbols (does not compare expressions!)
     *
     * @param sym1 First symbol
     * @param sym2 Second symbol
     *
     * @return `false` if symbols are equal, `true` otherwise
     */
    __host__ __device__ bool operator!=(const Symbol& sym1, const Symbol& sym2);

    /*
     * @brief Iterator of an array of `Symbol`s with capacity bounds checking.
     * This should be defined inside of `Symbol`, but forward declarations of nested classes are
     * impossible (and we need a forward declaration in Macros.cuh).
     */
    template <class S> class GenericSymbolIterator {
        S* parent = nullptr;
        size_t index_ = 0;
        size_t capacity_ = 0;

      public:
        /*
         * @brief Creates an iterator to a symbol sequence
         *
         * @param parent Parent expression of the expression
         * @param index Initial index of the iterator
         * @param capacity Capacity of the parent expression
         *
         * @return Good with the iterator on success, error when the given index is past the
         * allocated space of the parent
         */
        [[nodiscard]] __host__ __device__ static Util::SimpleResult<GenericSymbolIterator>
        from_at(S& parent, const size_t index, const size_t capacity) {
            GenericSymbolIterator iterator;
            iterator.parent = &parent;
            iterator.index_ = index;
            iterator.capacity_ = capacity;

            if (index >= capacity) {
                return Util::SimpleResult<GenericSymbolIterator>::make_error();
            }

            return Util::SimpleResult<GenericSymbolIterator>::make_good(iterator);
        }

        /*
         * @brief Index of the expression the iterator is pointing at
         */
        [[nodiscard]] __host__ __device__ size_t index() const { return index_; }

        /*
         * @brief Capacity of the iterated array
         */
        [[nodiscard]] __host__ __device__ size_t total_capacity() const { return capacity_; }

        /*
         * @brief How many `Symbol`s are within bounds starting from the current symbol (e.g. if
         * the current symbol is the last one, returns 1).
         *
         */
        [[nodiscard]] __host__ __device__ size_t capacity() const { return capacity_ - index_; }

        /*
         * @brief Const symbol pointed to by the iterator
         */
        [[nodiscard]] __host__ __device__ const Symbol& const_current() const {
            return parent[index_];
        };

        /*
         * @brief Symbol pointed to by the iterator
         */
        [[nodiscard]] __host__ __device__ Symbol& current() {
            return const_cast<Symbol&>(this->const_current());
        };

        [[nodiscard]] __host__ __device__ const Symbol& operator*() const {
            return const_current();
        };

        [[nodiscard]] __host__ __device__ Symbol& operator*() {
            return const_cast<Symbol&>(*const_cast<const GenericSymbolIterator<S>&>(*this));
        };

        [[nodiscard]] __host__ __device__ const Symbol* operator->() const { return &**this; };

        [[nodiscard]] __host__ __device__ Symbol* operator->() { return &**this; };

        /*
         * @brief `true` when the iterator can be offset by `offset` without going into
         * unallocated memory
         */
        template <class O>
        [[nodiscard]] __host__ __device__ bool can_offset_by(const O offset) const {
            return index_ + offset < capacity_;
        }

        /*
         * @brief Creates a new iterator offset by `offset` expressions
         *
         * @return Good with the iterator on success, error when the given index is past the
         * allocated space of the parent
         */
        template <class O>
        [[nodiscard]] __host__ __device__ Util::SimpleResult<GenericSymbolIterator>
        operator+(const O offset) const {
            return from_at(parent, index_ + offset);
        }

        /*
         * @brief Creates a new iterator offset by `-offset` expressions
         */
        template <class O>
        [[nodiscard]] __host__ __device__ Util::SimpleResult<GenericSymbolIterator>
        operator-(const O offset) const {
            return *this + (-offset);
        }

        /*
         * @brief Offsets the iterator by `offset`. Returns error when the resulting iterator
         * would point a location past the allocated space.
         */
        template <class O>
        [[nodiscard]] __host__ __device__ Util::BinaryResult operator+=(const O offset) {
            if (!can_offset_by(offset)) {
                return Util::BinaryResult::make_error();
            }

            index_ += offset;
            return Util::BinaryResult::make_good();
        }

        /*
         * @brief Offsets the iterator by `-offset. Returns error when the resulting iterator
         * would point a location past the allocated space.
         */
        template <class O>
        [[nodiscard]] __host__ __device__ Util::BinaryResult operator-=(const O offset) {
            return *this += -offset;
        }
    };

    using SymbolIterator = GenericSymbolIterator<Symbol>;
    using SymbolConstIterator = GenericSymbolIterator<const Symbol>;
}

#endif
