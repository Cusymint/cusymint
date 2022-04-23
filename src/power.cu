#include "power.cuh"

#include "symbol.cuh"

namespace Sym {
    std::string Power::to_string() {
        Symbol* sym = Symbol::from(this);
        return "(" + sym[1].to_string() + "^" + sym[second_arg_offset].to_string() + ")";
    }

    std::vector<Symbol> operator^(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs) {
        std::vector<Symbol> res(lhs.size() + rhs.size() + 1);
        res[0].power = Power::create();
        res[0].power.total_size = res.size();
        res[0].power.second_arg_offset = 1 + lhs.size();
        auto next = std::copy(lhs.begin(), lhs.end(), res.begin() + 1);
        std::copy(rhs.begin(), rhs.end(), next);

        return res;
    }
}
