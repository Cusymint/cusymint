#include "ErrorFunction.cuh"

#include <fmt/core.h>

#include "Symbol.cuh"
#include "Symbol/MetaOperators.cuh"

namespace Sym {
    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(ErrorFunction)
    DEFINE_SIMPLE_ONE_ARGUMENT_OP_ARE_EQUAL(ErrorFunction)
    DEFINE_IDENTICAL_COMPARE_TO(ErrorFunction)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(ErrorFunction)
    DEFINE_SIMPLE_ONE_ARGUMENT_IS_FUNCTION_OF(ErrorFunction)
    DEFINE_ONE_ARG_OP_DERIVATIVE(ErrorFunction, (Prod<Integer<2>, Inv<Sqrt<Pi>>, Pow<E, Neg<Pow<Copy, Integer<2>>>>>))
    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(ErrorFunction)

    [[nodiscard]] std::string ErrorFunction::to_string() const {
        return fmt::format("erf({})", arg().to_string());
    }

    [[nodiscard]] std::string ErrorFunction::to_tex() const {
        return fmt::format(R"(\text{{erf}}\left({}\right))", arg().to_tex());
    }

    std::vector<Symbol> erf(const std::vector<Symbol>& arg) {
        std::vector<Symbol> res(arg.size() + 1);
        ErrorFunction::create(arg.data(), res.data());
        return res;
    }
}