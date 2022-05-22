#include "product.cuh"

#include "symbol.cuh"

namespace Sym {
    DEFINE_TWO_ARGUMENT_OP_FUNCTIONS(Product)
    DEFINE_SIMPLE_TWO_ARGUMENT_OP_COMPARE(Product)
    DEFINE_TWO_ARGUMENT_OP_COMPRESS_REVERSE_TO(Product)

    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(Reciprocal)
    DEFINE_SIMPLE_ONE_ARGUMETN_OP_COMPARE(Reciprocal)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Reciprocal)

    std::string Product::to_string() const {
        if (arg1().is(Type::Reciprocal)) {
            return "(" + arg2().to_string() + "/" + arg1().reciprocal.arg().to_string() + ")";
        }
        else if (arg2().is(Type::Reciprocal)) {
            return "(" + arg1().to_string() + "/" + arg2().reciprocal.arg().to_string() + ")";
        }
        else {
            return "(" + arg1().to_string() + "*" + arg2().to_string() + ")";
        }
    }

    std::string Reciprocal::to_string() const { return "(1/" + arg().to_string() + ")"; }

    std::vector<Symbol> operator*(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs) {
        std::vector<Symbol> res(lhs.size() + rhs.size() + 1);
        Product::create(lhs.data(), rhs.data(), res.data());
        return res;
    }

    std::vector<Symbol> operator/(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs) {
        std::vector<Symbol> res(rhs.size() + 1);

        Reciprocal* const reciprocal = res.data() << Reciprocal::builder();
        rhs.data()->copy_to(&reciprocal->arg());
        reciprocal->seal();

        return lhs * res;
    }
}
