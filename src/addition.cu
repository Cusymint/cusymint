
#include "addition.cuh"

#include "symbol.cuh"

namespace Sym {
    DEFINE_SIMPLE_ONE_ARGUMETN_OP_COMPARE(Addition)
    DEFINE_TWO_ARGUMENT_OP_COMPRESS_REVERSE_TO(Addition)

    DEFINE_SIMPLE_ONE_ARGUMETN_OP_COMPARE(Negative)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Negative)

    std::string Addition::to_string() const {
        const Symbol* const sym = Symbol::from(this);
        if (sym[second_arg_offset].is(Type::Negative)) {
            return "(" + sym[1].to_string() + "-" + sym[second_arg_offset + 1].to_string() + ")";
        }
        else if (sym[second_arg_offset].is(Type::NumericConstant) &&
                 sym[second_arg_offset].numeric_constant.value < 0.0) {
            return "(" + sym[1].to_string() + "-" +
                   std::to_string(-sym[second_arg_offset].numeric_constant.value) + ")";
        }
        else {
            return "(" + sym[1].to_string() + "+" + sym[second_arg_offset].to_string() + ")";
        }
    }

    std::string Negative::to_string() const { return "-" + Symbol::from(this)[1].to_string(); }

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
}
