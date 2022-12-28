#include "MetaOperators.cuh"
#include "Sign.cuh"
#include "Symbol.cuh"

namespace Sym {
    DEFINE_ONE_ARGUMENT_OP_FUNCTIONS(Sign)
    DEFINE_SIMPLE_ONE_ARGUMENT_OP_ARE_EQUAL(Sign)
    DEFINE_IDENTICAL_COMPARE_TO(Sign)
    DEFINE_ONE_ARGUMENT_OP_COMPRESS_REVERSE_TO(Sign)
    DEFINE_SIMPLE_ONE_ARGUMENT_IS_FUNCTION_OF(Sign)

    DEFINE_INSERT_REVERSED_DERIVATIVE_AT(Sign) {
        destination.init_from(NumericConstant::with_value(0));
        return 1;
    }

    DEFINE_DERIVATIVE_SIZE(Sign) { return 1; }

    DEFINE_SIMPLIFY_IN_PLACE(Sign) {
        if (arg().is(Type::Negation)) {
            arg().as<Negation>().type = Type::Sign;
            type = Type::Negation;
        }

        if (arg().is(Type::Product)) {
            // This is done under assumptions that the product tree contains at most one
            // NumericConstant (because `arg()`  is simplified)
            for (TreeIterator<Product> iterator(&arg()); iterator.is_valid(); iterator.advance()) {
                if (!iterator.current()->is(Type::NumericConstant)) {
                    continue;
                }

                if (iterator.current()->as<NumericConstant>().value < 0) {
                    iterator.current()->as<NumericConstant>().value *= -1;

                    Neg<Copy>::init(*help_space, {symbol()});
                    help_space->copy_to(symbol());
                }

                break;
            }
        }

        if (arg().is(Type::NumericConstant)) {
            double value = 0.0;

            if (arg().as<NumericConstant>().value > 0.0) {
                value = 1.0;
            }
            else if (arg().as<NumericConstant>().value < 0.0) {
                value = -1.0;
            }

            symbol().init_from(NumericConstant::with_value(value));

            return true;
        }

        if (arg().is(Type::KnownConstant)) {
            double value = 0.0;

            switch (arg().as<KnownConstant>().value) {
            case KnownConstantValue::Unknown:
                return true;
            case KnownConstantValue::E:
                [[fallthrough]];
            case KnownConstantValue::Pi:
                value = 1.0;
                break;
            }

            symbol().init_from(NumericConstant::with_value(value));

            return true;
        }

        return true;
    }

    std::string Sign::to_string() const { return fmt::format("sgn({})", arg().to_string()); }

    std::string Sign::to_tex() const {
        return fmt::format(R"(\text{{sgn}}\left({}\right))", arg().to_tex());
    }

    std::vector<Symbol> sgn(const std::vector<Symbol>& arg) {
        std::vector<Symbol> res(arg.size() + 1);
        Sign::create(arg.data(), res.data());
        return res;
    }

    std::vector<Symbol> abs(const std::vector<Symbol>& arg) {
        using Abs = Mul<Sgn<Copy>, Copy>;
        Abs::AdditionalArgs args = {*arg.data(), *arg.data()};

        std::vector<Symbol> res(Abs::size_with(args));
        Abs::init(*res.data(), args);

        return res;
    }
}
