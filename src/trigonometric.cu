#include "trigonometric.cuh"

#include "symbol.cuh"

namespace Sym {
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
