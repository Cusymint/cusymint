
#include "addition.cuh"

#include "symbol.cuh"

namespace Sym {
    std::vector<Symbol> operator+(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs) {
        std::vector<Symbol> res(lhs.size() + rhs.size() + 1);
        res[0].addition = Addition::create();
        res[0].addition.total_size = res.size();
        res[0].addition.second_arg_offset = 1 + lhs.size();
        auto next = std::copy(lhs.begin(), lhs.end(), res.begin() + 1);
        std::copy(rhs.begin(), rhs.end(), next);

        return res;
    }

    std::vector<Symbol> operator-(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs) {
        std::vector<Symbol> res(lhs.size() + rhs.size() + 2);
        res[0].addition = Addition::create();
        res[0].addition.total_size = res.size();
        res[0].addition.second_arg_offset = 1 + lhs.size();
        auto next = std::copy(lhs.begin(), lhs.end(), res.begin() + 1);
        next->negative = Negative::create();
        next->negative.total_size = res.size();
        std::copy(rhs.begin(), rhs.end(), next + 1);

        return res;
    }

    std::vector<Symbol> operator-(const std::vector<Symbol>& arg) {
        std::vector<Symbol> res(arg.size() + 1);
        res[0].negative = Negative::create();
        res[0].negative.total_size = res.size();
        std::copy(arg.begin(), arg.end(), res.begin() + 1);

        return res;
    }
} // namespace Sym
