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

// Also works when `_member_function` returns void
#define VIRTUAL_CALL(_instance, _member_function, ...)                                     \
    (([&]() {                                                                              \
        switch ((_instance).unknown.type) {                                                \
        case Type::Symbol:                                                                 \
            Util::crash("Trying to access a virtual function (%s) on a pure Symbol",       \
                        #_member_function);                                                \
            break;                                                                         \
        case Type::NumericConstant:                                                        \
            return (_instance).numeric_constant._member_function(__VA_ARGS__);             \
        case Type::KnownConstant:                                                          \
            return (_instance).known_constant._member_function(__VA_ARGS__);               \
        case Type::UnknownConstant:                                                        \
            return (_instance).unknown_constant._member_function(__VA_ARGS__);             \
        case Type::Variable:                                                               \
            return (_instance).variable._member_function(__VA_ARGS__);                     \
        case Type::ExpanderPlaceholder:                                                    \
            return (_instance).expander_placeholder._member_function(__VA_ARGS__);         \
        case Type::SubexpressionCandidate:                                                 \
            return (_instance).subexpression_candidate._member_function(__VA_ARGS__);      \
        case Type::SubexpressionVacancy:                                                   \
            return (_instance).subexpression_vacancy._member_function(__VA_ARGS__);        \
        case Type::Integral:                                                               \
            return (_instance).integral._member_function(__VA_ARGS__);                     \
        case Type::Solution:                                                               \
            return (_instance).solution._member_function(__VA_ARGS__);                     \
        case Type::Substitution:                                                           \
            return (_instance).substitution._member_function(__VA_ARGS__);                 \
        case Type::Addition:                                                               \
            return (_instance).addition._member_function(__VA_ARGS__);                     \
        case Type::Negation:                                                               \
            return (_instance).negation._member_function(__VA_ARGS__);                     \
        case Type::Product:                                                                \
            return (_instance).product._member_function(__VA_ARGS__);                      \
        case Type::Reciprocal:                                                             \
            return (_instance).reciprocal._member_function(__VA_ARGS__);                   \
        case Type::Power:                                                                  \
            return (_instance).power._member_function(__VA_ARGS__);                        \
        case Type::Sine:                                                                   \
            return (_instance).sine._member_function(__VA_ARGS__);                         \
        case Type::Cosine:                                                                 \
            return (_instance).cosine._member_function(__VA_ARGS__);                       \
        case Type::Tangent:                                                                \
            return (_instance).tangent._member_function(__VA_ARGS__);                      \
        case Type::Cotangent:                                                              \
            return (_instance).cotangent._member_function(__VA_ARGS__);                    \
        case Type::Arcsine:                                                                \
            return (_instance).arcsine._member_function(__VA_ARGS__);                      \
        case Type::Arccosine:                                                              \
            return (_instance).arccosine._member_function(__VA_ARGS__);                    \
        case Type::Arctangent:                                                             \
            return (_instance).arctangent._member_function(__VA_ARGS__);                   \
        case Type::Arccotangent:                                                           \
            return (_instance).arccotangent._member_function(__VA_ARGS__);                 \
        case Type::Logarithm:                                                              \
            return (_instance).logarithm._member_function(__VA_ARGS__);                    \
        case Type::Polynomial:                                                             \
            return (_instance).polynomial._member_function(__VA_ARGS__);                   \
        case Type::Unknown:                                                                \
            return (_instance).unknown._member_function(__VA_ARGS__);                      \
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

        template <class T> [[nodiscard]] __host__ __device__ inline bool is() const {
            return type() == T::TYPE;
        }

        [[nodiscard]] __host__ __device__ inline size_t& size() { return unknown.size; }

        [[nodiscard]] __host__ __device__ inline size_t size() const { return unknown.size; }

        [[nodiscard]] __host__ __device__ inline size_t& capacity() { return unknown.capacity; }

        [[nodiscard]] __host__ __device__ inline size_t capacity() const {
            return unknown.capacity;
        }

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
         * @brief Pointer to the `idx`th element after `this`. Error if the index is out of range of
         * the `Symbol`'s capacity.
         */
        [[nodiscard]] __host__ __device__ inline Util::SimpleResult<const Symbol*>
        operator[](const size_t idx) const {
            if (capacity() <= idx) {
                return Util::SimpleResult<const Symbol*>::error();
            }

            return Util::SimpleResult<const Symbol*>::good(this + idx);
        }

        /*
         * @brief Pointer to the `idx`th element after `this`. Error if the index is out of range of
         * the `Symbol`'s capacity.
         */
        [[nodiscard]] __host__ __device__ inline Util::SimpleResult<Symbol*>
        operator[](const size_t idx) {
            return const_cast<const Symbol*>(this)->operator[](idx).map_good<Symbol*>(
                [](const Symbol* const good) {
                    return Util::SimpleResult<Symbol*>::good(const_cast<Symbol*>(good));
                });
        }

        /*
         * @brief Reference to the `idx`th element after `this`
         */
        [[nodiscard]] __host__ __device__ inline const Symbol&
        at_unchecked(const size_t idx) const {
            if constexpr (Consts::DEBUG) {
                // If `this` is under construction, we allow access to symbols after it without
                // checks
                if (capacity() != BUILDER_SIZE && capacity() <= idx) {
                    Util::crash("Trying to access element after a Symbol at index %lu without "
                                "checking bounds, but the symbol's "
                                "capacity is %lu and it is not under construction",
                                idx, capacity());
                }
            }

            return *(this + idx);
        }

        /*
         * @brief Reference to the `idx`th element after `this`
         */
        [[nodiscard]] __host__ __device__ inline Symbol& at_unchecked(const size_t idx) {
            return const_cast<Symbol&>(const_cast<Symbol* const>(this)->at_unchecked(idx));
        }

        /*
         * @brief Pointer to the symbol right after `this`
         */
        [[nodiscard]] __host__ __device__ inline const Symbol& child() const {
            return at_unchecked(1);
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
         * @brief Moves symbol sequence from `source` to `destination`. Source and destination can
         * alias.
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
        __host__ __device__ bool is_function_of(const Symbol* const* const expressions,
                                                const size_t expression_count) const;

        /*
         * @brief Removes holes from symbol tree and copies it in reverse order to
         * `destination`.
         *
         * @param destination Location to which the tree is going to be copied
         *
         * @return New size of the symbol tree
         */
        __host__ __device__ size_t compress_reverse_to(Symbol* const destination);

        /*
         * @brief Removes holes from symbol tree and copies it to `destination`.
         *
         * @param destination Location to which the tree is going to be copied. Cannot be same
         * as `this`.
         *
         * @return New size of the symbol tree
         */
        __host__ __device__ size_t compress_to(Symbol& destination);

        /*
         * @brief Zwraca funkcję podcałkową jeśli `this` jest całką. Undefined behavior w
         * przeciwnym wypadku.
         *
         * @return Wskaźnik do funkcji podcałkowej
         */
        [[nodiscard]] __host__ __device__ inline Symbol* integrand() {
            return integral.integrand();
        }
        [[nodiscard]] __host__ __device__ inline const Symbol* integrand() const {
            return integral.integrand();
        }

        __host__ __device__ void
        mark_to_be_copied_and_propagate_additional_size(Symbol* const help_space);

        /*
         * @brief Simplifies an expression
         *
         * @param help_space Help space
         */
        __host__ __device__ void simplify(Symbol* const help_space);

        /*
         * @brief Simplified an expression. Can leave the expression with holes.
         *
         * @param help_space Help space.
         *
         * @return `true` if expression was simplified, `false` if simplified result
         * would take more space than `size()` or expression needs to be simplified again.
         */
        __host__ __device__ bool simplify_in_place(Symbol* const help_space);

        /*
         * @brief Recalculates sizes and argument offsets in the given expression. There cannot be
         * any holes in the expression.
         *
         * @param expr Expression to seal
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
         * @brief If `this` is a monomial, returns its coefficient. Otherwise, returns `empty_num`.
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

        template <class S> class GenericIterator {
            S* parent = nullptr;
            size_t index_ = 0;

            GenericIterator() = default;

          public:
            /*
             * @brief Creates an iterator to a symbol sequence
             *
             * @return Good with the iterator on success, error when the given index is past the
             * allocated space of the parent
             */
            __host__ __device__ static Util::SimpleResult<GenericIterator>
            from_at(S& parent, const size_t index) {
                GenericIterator iterator;
                iterator.parent = parent;
                iterator.index_ = index;

                if (index >= parent.capacity()) {
                    return Util::SimpleResult<GenericIterator>::error();
                }

                return Util::SimpleResult<GenericIterator>::good(iterator);
            }

            /*
             * @brief Index of the expression the iterator is pointing at
             */
            [[nodiscard]] __host__ __device__ size_t index() const { return index_; }

            /*
             * @brief Const symbol pointed to by the iterator
             */
            [[nodiscard]] __host__ __device__ const Symbol* const_current() const {
                return parent->at_unchecked(index_);
            };

            /*
             * @brief Symbol pointed to by the iterator
             */
            [[nodiscard]] __host__ __device__ Symbol& current() {
                return const_cast<Symbol&>(this->const_current());
            };

            /*
             * @brief `true` when the iterator can be offset by `offset` without going into
             * unallocated memory
             */
            template <class O>
            [[nodiscard]] __host__ __device__ bool can_offset_by(const O offset) const {
                return index_ + offset < parent->capacity();
            }

            /*
             * @brief Creates a new iterator offset by `offset` expressions
             *
             * @return Good with the iterator on success, error when the given index is past the
             * allocated space of the parent
             */
            template <class O>
            [[nodiscard]] __host__ __device__ Util::SimpleResult<GenericIterator>
            operator+(const O offset) const {
                return from_at(parent, index_ + offset);
            }

            /*
             * @brief Creates a new iterator offset by `-offset` expressions
             */
            template <class O>
            [[nodiscard]] __host__ __device__ Util::SimpleResult<GenericIterator>
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
                    return Util::BinaryResult::error();
                }

                index_ += offset;
                return Util::BinaryResult::good();
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

        using Iterator = GenericIterator<Symbol>;
        using ConstIterator = GenericIterator<const Symbol>;
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
}

#endif
