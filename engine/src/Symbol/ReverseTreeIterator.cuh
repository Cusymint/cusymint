#ifndef REVERSE_TREE_ITERATOR_CUH
#define REVERSE_TREE_ITERATOR_CUH

#include "Symbol.cuh"

namespace Sym {
    /*
     * @brief Iterator of expressions in a simplified expression tree, giving items in reversed order
     */
    template <class T, class S> class GenericReverseTreeIterator {
        T* root;
        T* current_op;
        S* current_symbol;

      public:
        /*
         * @brief Creates a tree iterator
         *
         * @param tree Operator over which children the iterator will iterate
         */
        __host__ __device__ explicit GenericReverseTreeIterator(T* const tree) :
            root(tree), current_op(tree->last_in_tree()), current_symbol(&current_op->arg1()) {}

        /*
         * @brief Creates a tree iterator if tree points to an element of type T, otherwise creates
         * a dummy iterator with no current_op over a single expression
         *
         * @param tree Operator over which children the iterator will iterate
         */
        __host__ __device__ explicit GenericReverseTreeIterator(S* const expression) {
            if (expression->is(T::TYPE)) {
                root = expression->template as_ptr<T>();
                current_op = root->last_in_tree();
                current_symbol = &current_op->arg1();
            }
            else {
                current_op = nullptr;
                current_symbol = expression;
            }
        }

        /*
         * @brief Moves the iterator forward
         *
         * @return `true` if the iterator is still valid after the move, `false` otherwise
         */
        __host__ __device__ bool advance() {
            if (current_op == nullptr) {
                current_symbol = nullptr;
                return false;
            }

            if (current_symbol == &current_op->arg1()) {
                current_symbol = &current_op->arg2();
                return true;
            }

            if (current_op != root) {
                current_op = (current_op->symbol() - 1)->template as_ptr<T>();
                current_symbol = &current_op->arg2();
                return true;
            }

            current_symbol = nullptr;
            current_op = nullptr;
            return false;
        }

        /*
         * @brief Wheather the iterator is valid
         *
         * @return `true` if `current() != nullptr`, `false` otherwise
         */
        [[nodiscard]] __host__ __device__ bool is_valid() const {
            return current_symbol != nullptr;
        }

        /*
         * @brief Current element
         *
         * @return Element pointed to by the iterator. `nullptr` if the iterator has been exhaused.
         */
        [[nodiscard]] __host__ __device__ const Symbol* current() const {
            if constexpr (Consts::DEBUG) {
                if (!is_valid()) {
                    Util::crash("Trying to access the current element of an exhaused iterator");
                }
            }

            return current_symbol;
        }

        /*
         * @brief Current element
         *
         * @return Element pointed to by the iterator. `nullptr` if the iterator has been exhaused.
         */
        [[nodiscard]] __host__ __device__ Symbol* current() {
            return const_cast<Symbol*>(const_cast<const GenericReverseTreeIterator*>(this)->current());
        }
    };

    template <class T> using ReverseTreeIterator = GenericReverseTreeIterator<T, Symbol>;
    template <class T> using ConstReverseTreeIterator = GenericReverseTreeIterator<const T, const Symbol>;
}

#endif
