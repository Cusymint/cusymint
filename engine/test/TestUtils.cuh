#include "Symbol/Macros.cuh"
#include "Symbol/Symbol.cuh"

namespace Test {
    /*
     * @brief Contains the same algorithm as `Symbol::simplify`, but every iteration prints
     * TeX-formatted partial result. Useful for debugging on host.
     */
    Util::BinaryResult simplify_verbose(Sym::Symbol& symbol, Sym::SymbolIterator& help_space) {
        bool success = false;

        printf("%s \\\\\n\n", symbol.to_tex().c_str());

        while (!success) {
            success = true;

            for (ssize_t i = static_cast<ssize_t>(symbol.size()) - 1; i >= 0; --i) {
                success = symbol.at(i)->simplify_in_place(help_space) && success;
            }

            const size_t new_size = TRY_PASS(Util::Empty, symbol.compress_reverse_to(help_space));
            Sym::Symbol::copy_and_reverse_symbol_sequence(symbol, *help_space, new_size);

            printf("%s \\\\\n", symbol.to_tex().c_str());
        }

        return Util::BinaryResult::make_good();
    }
}