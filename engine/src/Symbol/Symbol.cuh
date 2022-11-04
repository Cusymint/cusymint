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
#include "Logarithm.cuh"

#include "Utils/CompileConstants.cuh"
#include "Utils/Cuda.cuh"

// Also works when `_member_function` returns void
#define VIRTUAL_CALL(_instance, _member_function, ...)                                     \
    (([&]() {                                                                              \
        switch ((_instance).unknown.type) {                                                \
        case Type::Symbol:                                                                 \
            Util::crash("Trying to access a virtual function (%s) on a pure Symbol",       \
                        #_member_function);                                                \
            break;                                                                         \
        case Type::Variable:                                                               \
            return (_instance).variable._member_function(__VA_ARGS__);                     \
        case Type::NumericConstant:                                                        \
            return (_instance).numeric_constant._member_function(__VA_ARGS__);             \
        case Type::KnownConstant:                                                          \
            return (_instance).known_constant._member_function(__VA_ARGS__);               \
        case Type::UnknownConstant:                                                        \
            return (_instance).unknown_constant._member_function(__VA_ARGS__);             \
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
        Variable variable;
        NumericConstant numeric_constant;
        KnownConstant known_constant;
        UnknownConstant unknown_constant;
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

        constexpr static Sym::Type TYPE = Sym::Type::Symbol;

        [[nodiscard]] __host__ __device__ inline Type type() const { return unknown.type; }
        [[nodiscard]] __host__ __device__ inline bool simplified() const {
            return unknown.simplified;
        }
        [[nodiscard]] __host__ __device__ inline bool is(const Type other_type) const {
            return type() == other_type;
        }

        template <class T> [[nodiscard]] __host__ __device__ inline bool is() const {
            return type() == T::TYPE;
        }

        [[nodiscard]] __host__ __device__ inline size_t& size() { return unknown.size; }
        [[nodiscard]] __host__ __device__ inline size_t size() const { return unknown.size; }

        [[nodiscard]] __host__ __device__ inline size_t& additional_required_size() { return unknown.additional_required_size; }
        [[nodiscard]] __host__ __device__ inline size_t additional_required_size() const { return unknown.additional_required_size; }

        [[nodiscard]] __host__ __device__ inline bool& to_be_copied() { return unknown.to_be_copied; }
        [[nodiscard]] __host__ __device__ inline bool to_be_copied() const { return unknown.to_be_copied; }

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
         * @brief Zwraca wskaźnik na n-ty element za `this`
         */
        [[nodiscard]] __host__ __device__ const inline Symbol* at(const size_t idx) const {
            if constexpr (Consts::DEBUG) {
                // If `this` is under construction, we allow access to symbols after it without
                // checks
                if (size() != BUILDER_SIZE && size() <= idx) {
                    Util::crash(
                        "Trying to access %lu element after a symbol, but the symbol's size is %lu",
                        idx, size());
                }
            }

            return this + idx;
        }

        /*
         * @brief Pointer to the nth element after `this`
         */
        [[nodiscard]] __host__ __device__ inline Symbol* at(const size_t idx) {
            return const_cast<Symbol*>(const_cast<const Symbol*>(this)->at(idx));
        }

        /*
         * @brief Zwraca n-ty element za `this`
         */
        [[nodiscard]] __host__ __device__ inline const Symbol& operator[](const size_t idx) const {
            return *at(idx);
        }

        /*
         * @brief Zwraca n-ty element za `this`
         */
        [[nodiscard]] __host__ __device__ inline Symbol& operator[](const size_t idx) {
            return *const_cast<Symbol*>(&(*const_cast<const Symbol*>(this))[idx]);
        }

        /*
         * @brief Zwraca wskaźnik na symbol bezpośrednio za `this`
         */
        [[nodiscard]] __host__ __device__ inline const Symbol* child() const { return at(1); }

        /*
         * @brief Zwraca wskaźnik na symbol bezpośrednio za `this`
         */
        [[nodiscard]] __host__ __device__ inline Symbol* child() {
            return const_cast<Symbol*>(const_cast<const Symbol*>(this)->child());
        }

        /*
         * @brief Copies symbol sequence from `source` to `destination`.
         *
         * @param seq Symbol sequence to copy. Doesn't need to be semantically correct.
         * @param n number of symbols to copy
         */
        __host__ __device__ static void copy_symbol_sequence(Symbol* const destination,
                                                             const Symbol* const source, size_t n);

        /*
         * @brief Copies symbol sequence from `source` to `destination` in reverse.
         *
         * @param seq Symbol sequence to copy. Doesn't need to be semantically correct.
         * @param n number of symbols to copy
         */
        __host__ __device__ static void copy_and_reverse_symbol_sequence(Symbol* const destination,
                                                                         const Symbol* const source,
                                                                         size_t n);

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
        __host__ __device__ void copy_single_to(Symbol* const destination) const;

        /*
         * @brief Copies `this` into `destination`. Copies the whole tree.
         *
         * @param destination Copy destination
         */
        __host__ __device__ void copy_to(Symbol* const destination) const;

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
         * expressed as `g1(x), g2(x), ..., gn(x)`, this function checks if there exists a function
         * `h(x1, x2, ..., xn)`, such that `f(x) = h(g1(x), g2(x), ..., gn(x))`.
         *
         * @param expressions Expressions to check
         *
         * @return `true` if `this` is a function of `expressions`, false otherwise. Returns false
         * also when `this` is constant. In particular, it returns true if `this` is a constant
         * expression
         */
        template <class... Args, std::enable_if_t<(std::is_same_v<Args, Symbol> && ...), int> = 0>
        [[nodiscard]] __host__ __device__ bool is_function_of(const Args&... expressions) const {
            const Symbol* const expression_array[] = {&expressions...};
            return VIRTUAL_CALL(*this, is_function_of, expression_array, sizeof...(Args));
        }

        /*
         * @brief Another version of `is_function_of` that is the same as the functions defined in
         * concrete symbols. It should be called only when the templated version cannot be used.
         *
         * @param expressions Array of pointers to expressions to check
         * @param expression_count Number of expressions in `expressions`
         *
         * @return Same as in the other function
         */
        __host__ __device__ bool is_function_of(const Symbol* const* const expressions,
                                                const size_t expression_count) const;

        /*
         * @brief Replaces every occurence of `expr` (which has to contain a variable) in `this`
         * with variable and copies the result to `destination`. If size of `expr` is larger
         * than 1, holes are left where symbols were before.
         *
         * @param destination Destination of copy
         * @param expr Expression to replace in `this`. Has to contain a variable.
         */
        __host__ __device__ void substitute_with_var_with_holes(Symbol& destination,
                                                                const Symbol& expression) const;

        /*
         * @brief Removes holes from symbol tree and copies it in reverse order to `destination`.
         *
         * @param destination Location to which the tree is going to be copied
         *
         * @return New size of the symbol tree
         */
        __host__ __device__ size_t compress_reverse_to(Symbol* const destination);

        /*
         * @brief Zwraca funkcję podcałkową jeśli `this` jest całką. Undefined behavior w przeciwnym
         * wypadku.
         *
         * @return Wskaźnik do funkcji podcałkowej
         */
        [[nodiscard]] __host__ __device__ inline Symbol* integrand() {
            return integral.integrand();
        }
        [[nodiscard]] __host__ __device__ inline const Symbol* integrand() const {
            return integral.integrand();
        }

        __host__ __device__ void mark_to_be_copied_and_propagate_additional_size(Symbol* const help_space);

        /*
         * @brief Wykonuje uproszcznie wyrażenia
         * Wykorzystuje założenie, że wyrażenie uproszczone jest krótsze od pierwotnego.
         *
         * @param help_space Pamięć pomocnicza
         */
        __host__ __device__ void simplify(Symbol* const help_space);

        /*
         * @brief Wykonuje uproszcznie wyrażenia, potencjalnie zostawiając dziury w wyrażeniu
         * Wykorzystuje założenie, że wyrażenie uproszczone jest krótsze od pierwotnego.
         *
         * @param help_space Pamięć pomocnicza
         *
         * @return `true` if expression was simplified, `false` if simplified result
         * would take more space than `size()` or expression needs to be simplified again.
         */
        __host__ __device__ bool simplify_in_place(Symbol* const help_space);

        /*
         * @brief Substitutes all occurences of variable with `symbol`
         *
         * @param symbol Symbol to substitute variables with, cannot have any children
         */
        void substitute_variable_with(const Symbol symbol);

        /*
         * @brief Wrapper do `substitute_variable_with` ułatwiający tworzenie stringów z
         * podstawieniami
         *
         * @param n Numer podstawienia z którego wziąta będzie nazwa
         */
        void substitute_variable_with_nth_substitution_name(const size_t n);

        /*
         * @brief Jeśli ten symbol jest typu T, to wykonaj na nim `function`
         *
         * @param function Funkcja która się wykona jeśli `this` jest typu T. Jako argument podawane
         * jest `this`
         */
        template <class T, class F> __host__ __device__ void if_is_do(F function) {
            if (T::TYPE == type()) {
                function(as<T>());
            }
        }

        /*
         * @brief Jeśli ten symbol jest typu T, to wykonaj na nim `function`
         *
         * @param function Funkcja która się wykona jeśli `this` jest typu T. Jako argument podawane
         * jest `this`
         */
        template <class T, class F> __host__ __device__ void if_is_do(F function) const {
            if (T::TYPE == type()) {
                function(as<T>());
            }
        }

        /*
         * @brief Porównuje dwa drzewa wyrażeń, oba muszą być uproszczone.
         *
         * @param expr1 Pierwsze wyrażenie
         * @param expr2 Drugie wyrażenie
         *
         * @return `true` jeśli drzewa wyrażeń mają tę samą strukturę, `false` w przeciwnym wypadku
         */
        __host__ __device__ static bool compare_trees(const Symbol* const expr1,
                                                      const Symbol* const expr2);

        /*
         * @brief Zwraca przyjazny użytkownikowi `std::string` reprezentujący wyrażenie.
         */
        [[nodiscard]] std::string to_string() const;

        /*
         * @brief Zwraca zapis wyrażenia w formacie TeX-a.
         */
        [[nodiscard]] std::string to_tex() const;
    };

    /*
     * @brief porównanie pojedynczego symbolu (bez porównywania drzew)
     *
     * @param sym1 pierwszy symbol
     * @param sym2 drugi symbol
     *
     * @return `true` jeśli symbole są równe, `false` jeśli nie
     */
    __host__ __device__ bool operator==(const Symbol& sym1, const Symbol& sym2);

    /*
     * @brief porównanie pojedynczych symbolu (bez porównywania drzew)
     *
     * @param sym1 pierwszy symbol
     * @param sym2 drugi symbol
     *
     * @return `true` jeśli symbole nie są równe, `false` jeśli są
     */
    __host__ __device__ bool operator!=(const Symbol& sym1, const Symbol& sym2);
}

#endif
