#include "IntegralFunctions.cuh"

#include <fmt/core.h>

#include "Symbol.cuh"
#include "Symbol/Macros.cuh"
#include "Symbol/MetaOperators.cuh"

namespace Sym {
    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(SineIntegral)
    DEFINE_SIMPLE_ONE_ARGUMENT_OP_ARE_EQUAL(SineIntegral)
    DEFINE_IDENTICAL_COMPARE_TO(SineIntegral)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(SineIntegral)
    DEFINE_SIMPLE_ONE_ARGUMENT_IS_FUNCTION_OF(SineIntegral)
    DEFINE_INSERT_REVERSED_DERIVATIVE_AT(SineIntegral) {
        if ((destination - 1)->is(0)) {
            return 0;
        }
        return Mul<Frac<Sin<Copy>, Copy>, None>::init_reverse(*destination, {arg(), arg()});
    }
    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(SineIntegral)

    [[nodiscard]] std::string SineIntegral::to_string() const {
        return fmt::format("Si({})", arg().to_string());
    }

    [[nodiscard]] std::string SineIntegral::to_tex() const {
        return fmt::format(R"(\text{{Si}}\left({}\right))", arg().to_tex());
    }

    std::vector<Symbol> si(const std::vector<Symbol>& arg) {
        std::vector<Symbol> res(arg.size() + 1);
        SineIntegral::create(arg.data(), res.data());
        return res;
    }

    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(CosineIntegral)
    DEFINE_SIMPLE_ONE_ARGUMENT_OP_ARE_EQUAL(CosineIntegral)
    DEFINE_IDENTICAL_COMPARE_TO(CosineIntegral)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(CosineIntegral)
    DEFINE_SIMPLE_ONE_ARGUMENT_IS_FUNCTION_OF(CosineIntegral)
    DEFINE_INSERT_REVERSED_DERIVATIVE_AT(CosineIntegral) {
        if ((destination - 1)->is(0)) {
            return 0;
        }
        return Mul<Frac<Cos<Copy>, Copy>, None>::init_reverse(*destination, {arg(), arg()});
    }
    DEFINE_NO_OP_SIMPLIFY_IN_PLACE(CosineIntegral)

    [[nodiscard]] std::string CosineIntegral::to_string() const {
        return fmt::format("Ci({})", arg().to_string());
    }

    [[nodiscard]] std::string CosineIntegral::to_tex() const {
        return fmt::format(R"(\text{{Ci}}\left({}\right))", arg().to_tex());
    }

    std::vector<Symbol> ci(const std::vector<Symbol>& arg) {
        std::vector<Symbol> res(arg.size() + 1);
        CosineIntegral::create(arg.data(), res.data());
        return res;
    }

    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(ExponentialIntegral)
    DEFINE_SIMPLE_ONE_ARGUMENT_OP_ARE_EQUAL(ExponentialIntegral)
    DEFINE_IDENTICAL_COMPARE_TO(ExponentialIntegral)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(ExponentialIntegral)
    DEFINE_SIMPLE_ONE_ARGUMENT_IS_FUNCTION_OF(ExponentialIntegral)
    DEFINE_INSERT_REVERSED_DERIVATIVE_AT(ExponentialIntegral){
        if ((destination - 1)->is(0)) {
            return 0;
        }
        return Mul<Frac<Pow<E, Copy>, Copy>, None>::init_reverse(*destination, {arg(), arg()});
    }
    
    DEFINE_SIMPLIFY_IN_PLACE(ExponentialIntegral) {
        if (arg().is(Type::Logarithm)) {
            arg().as<Logarithm>().arg().move_to(arg());
            type = Type::LogarithmicIntegral;
            seal();
        }
        return true;
    }

    [[nodiscard]] std::string ExponentialIntegral::to_string() const {
        return fmt::format("Ei({})", arg().to_string());
    }

    [[nodiscard]] std::string ExponentialIntegral::to_tex() const {
        return fmt::format(R"(\text{{Ei}}\left({}\right))", arg().to_tex());
    }

    std::vector<Symbol> ei(const std::vector<Symbol>& arg) {
        std::vector<Symbol> res(arg.size() + 1);
        ExponentialIntegral::create(arg.data(), res.data());
        return res;
    }

    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(LogarithmicIntegral)
    DEFINE_SIMPLE_ONE_ARGUMENT_OP_ARE_EQUAL(LogarithmicIntegral)
    DEFINE_IDENTICAL_COMPARE_TO(LogarithmicIntegral)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(LogarithmicIntegral)
    DEFINE_SIMPLE_ONE_ARGUMENT_IS_FUNCTION_OF(LogarithmicIntegral)
    DEFINE_ONE_ARG_OP_DERIVATIVE(LogarithmicIntegral, Inv<Ln<Copy>>)
    
    DEFINE_SIMPLIFY_IN_PLACE(LogarithmicIntegral) {
        if (Pow<E, Any>::match(arg())) {
            arg().as<Power>().arg2().move_to(arg());
            type = Type::ExponentialIntegral;
            seal();
        }
        return true;
    }

    [[nodiscard]] std::string LogarithmicIntegral::to_string() const {
        return fmt::format("li({})", arg().to_string());
    }

    [[nodiscard]] std::string LogarithmicIntegral::to_tex() const {
        return fmt::format(R"(\text{{li}}\left({}\right))", arg().to_tex());
    }

    std::vector<Symbol> li(const std::vector<Symbol>& arg) {
        std::vector<Symbol> res(arg.size() + 1);
        LogarithmicIntegral::create(arg.data(), res.data());
        return res;
    }
}