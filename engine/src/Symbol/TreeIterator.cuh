#ifndef TREE_ITERATOR_HPP
#define TREE_ITERATOR_HPP

#include "Symbol.cuh"

namespace Sym {
    /*
     * @brief Iterator kolejnych wyrażeń w uproszczonym drzewie operatora
     */
    template <class T, class S> class GenericTreeIterator {
        T* current_op;
        S* current_symbol;

      public:
        /*
         * @brief Tworzy iterator dla drzewa wyrażenia
         *
         * @param op Symbol po którego dzieciach miejsce będzie mieć iteracja
         */
        __host__ __device__ explicit GenericTreeIterator(T* const tree) :
            current_op(tree), current_symbol(&tree->arg2()) {}

        /*
         * @brief Przesuwa iterator do przodu.
         *
         * @return `true` jeśli przesunięty na kolejny element, `false` w przeciwnym wypadku.
         */
        __host__ __device__ bool advance() {
            if (current_op->arg1().is(T::TYPE)) {
                current_op = &current_op->arg1().template as<T>();
                current_symbol = &current_op->arg2();
                return true;
            }

            if (current_symbol == &current_op->arg2()) {
                current_symbol = &current_op->arg1();
                return true;
            }

            current_symbol = nullptr;
            current_op = nullptr;
            return false;
        }

        /*
         * @brief Zwraca informacje o ważności iteratora.
         *
         * @return `true` jeśli `current() != nullptr`, `false` w przeciwnym wypadku.
         */
        [[nodiscard]] __host__ __device__ bool is_valid() const {
            return current_symbol != nullptr;
        }

        /*
         * @brief Zwraca obecny element
         *
         * @return Element na który obecnie wskazuje iterator. Zwraca `nullptr` gdy koniec.
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
         * @brief Zwraca obecny element
         *
         * @return Element na który obecnie wskazuje iterator. Zwraca `nullptr` gdy koniec.
         */
        [[nodiscard]] __host__ __device__ Symbol* current() {
            return const_cast<Symbol*>(const_cast<const GenericTreeIterator*>(this)->current());
        }
    };

    template <class T> using TreeIterator = GenericTreeIterator<T, Symbol>;
    template <class T> using ConstTreeIterator = GenericTreeIterator<const T, const Symbol>;
}

#endif
