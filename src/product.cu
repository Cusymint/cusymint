#include "product.cuh"

#include "symbol.cuh"

namespace Sym {
    std::string Product::to_string() {
        Symbol* sym = Symbol::from(this);
        if (sym[1].is(Type::Reciprocal)) {
            return "(" + sym[second_arg_offset].to_string() + "/" + sym[2].to_string() + ")";
        }
        else if (sym[second_arg_offset].is(Type::Reciprocal)) {
            return "(" + sym[1].to_string() + "/" + sym[second_arg_offset + 1].to_string() + ")";
        }
        else {
            return "(" + sym[1].to_string() + "*" + sym[second_arg_offset].to_string() + ")";
        }
    }

    std::string Reciprocal::to_string() { return "(1/" + Symbol::from(this)[1].to_string() + ")"; }

    std::vector<Symbol> operator*(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs) {
        std::vector<Symbol> res(lhs.size() + rhs.size() + 1);
        res[0].product = Product::create();
        res[0].product.total_size = res.size();
        res[0].product.second_arg_offset = 1 + lhs.size();
        auto next = std::copy(lhs.begin(), lhs.end(), res.begin() + 1);
        std::copy(rhs.begin(), rhs.end(), next);

        return res;
    }

    std::vector<Symbol> operator/(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs) {
        std::vector<Symbol> res(lhs.size() + rhs.size() + 2);
        res[0].product = Product::create();
        res[0].product.total_size = res.size();
        res[0].product.second_arg_offset = 1 + lhs.size();
        auto next = std::copy(lhs.begin(), lhs.end(), res.begin() + 1);
        next->reciprocal = Reciprocal::create();
        next->reciprocal.total_size = res.size();
        std::copy(rhs.begin(), rhs.end(), next + 1);

        return res;
    }
}
