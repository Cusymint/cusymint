#include "trigonometric.cuh"

#include "symbol.cuh"

namespace Sym {
    DEFINE_SIMPLE_ONE_ARGUMETN_OP_COMPARE(Sine)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Sine)

    DEFINE_SIMPLE_ONE_ARGUMETN_OP_COMPARE(Cosine)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Cosine)

    DEFINE_SIMPLE_ONE_ARGUMETN_OP_COMPARE(Tangent)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Tangent)

    DEFINE_SIMPLE_ONE_ARGUMETN_OP_COMPARE(Cotangent)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Cotangent)

    std::string Sine::to_string() const { return "sin(" + Symbol::from(this)[1].to_string() + ")"; }

    std::string Cosine::to_string() const {
        return "cos(" + Symbol::from(this)[1].to_string() + ")";
    }

    std::string Tangent::to_string() const {
        return "tan(" + Symbol::from(this)[1].to_string() + ")";
    }

    std::string Cotangent::to_string() const {
        return "cot(" + Symbol::from(this)[1].to_string() + ")";
    }

    std::vector<Symbol> sin(const std::vector<Symbol>& arg) {
        std::vector<Symbol> res(arg.size() + 1);
        res[0].sine = Sine::create();
        res[0].sine.total_size = res.size();
        std::copy(arg.begin(), arg.end(), res.begin() + 1);

        return res;
    }

    std::vector<Symbol> cos(const std::vector<Symbol>& arg) {
        std::vector<Symbol> res(arg.size() + 1);
        res[0].cosine = Cosine::create();
        res[0].cosine.total_size = res.size();
        std::copy(arg.begin(), arg.end(), res.begin() + 1);

        return res;
    }

    std::vector<Symbol> tan(const std::vector<Symbol>& arg) {
        std::vector<Symbol> res(arg.size() + 1);
        res[0].tangent = Tangent::create();
        res[0].tangent.total_size = res.size();
        std::copy(arg.begin(), arg.end(), res.begin() + 1);

        return res;
    }

    std::vector<Symbol> cot(const std::vector<Symbol>& arg) {
        std::vector<Symbol> res(arg.size() + 1);
        res[0].cotangent = Cotangent::create();
        res[0].cotangent.total_size = res.size();
        std::copy(arg.begin(), arg.end(), res.begin() + 1);

        return res;
    }
}
