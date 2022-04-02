#include "trigonometric.cuh"

#include "symbol.cuh"

namespace Sym {
    std::string Sine::to_string() {
        return "sin(" + Symbol::from(this)[1].to_string() + ")";
    }

    std::string Cosine::to_string() {
        return "cos(" + Symbol::from(this)[1].to_string() + ")";
    }

    std::string Tangent::to_string() {
        return "tan(" + Symbol::from(this)[1].to_string() + ")";
    }

    std::string Cotangent::to_string() {
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
