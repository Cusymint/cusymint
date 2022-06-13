#ifndef TREE_ITERATOR_HPP
#define TREE_ITERATOR_HPP

#include "symbol.cuh"

namespace Sym {
    /*
     * @brief Iterator kolejnych wyrażeń w uproszczonym drzewie operatora
     */
    template <class T, Type t> class TreeIterator {
        T* current_op;
        Symbol* current_symbol;

      public:
        /*
         * @brief Tworzy iterator dla `operator`
         *
         * @param op Symbol po którego dzieciach miejsce będzie mieć iteracja
         */
        __host__ __device__ TreeIterator(T* const op)
            : current_op(op), current_symbol(&op->arg2()) {}

        /*
         * @brief Przesuwa iterator do przodu.
         *
         * @return `true` jeśli przesunięty na kolejny element, `false` w przeciwnym wypadku.
         */
        __host__ __device__ bool advance() {
            if (current_op->arg1().is(t)) {
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
        __host__ __device__ bool is_valid() { return current_symbol != nullptr; }

        /*
         * @brief Zwraca obecny element
         *
         * @return Element na który obecnie wskazuje iterator. Zwraca `nullptr` gdy koniec.
         */
        __host__ __device__ Symbol* current() { return current_symbol; }
    };

}

#endif
