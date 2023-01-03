#include "MetaOperators.cuh"
#include "Sign.cuh"
#include "Symbol.cuh"

namespace Sym {
    namespace {
        // This function checks only some simple cases, a complete check would require full-blown
        // inequality solver
        __host__ __device__ bool is_always_positive(const Symbol& expr) {
            if (expr.is(Type::NumericConstant)) {
                return expr.as<NumericConstant>().value > 0.0;
            }

            if (expr.is(Type::KnownConstant)) {
                switch (expr.as<KnownConstant>().value) {
                case KnownConstantValue::Unknown:
                    return false;
                case KnownConstantValue::E:
                    [[fallthrough]];
                case KnownConstantValue::Pi:
                    return true;
                }
            }

            if (Pow<Any, Num>::match(expr)) {
                const double& exponent = expr.as<Power>().arg2().as<NumericConstant>().value;

                if (floor(exponent) == exponent && static_cast<int64_t>(exponent) % 2 == 0) {
                    return true;
                }
            }

            return false;
        }
    }

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
        if (arg().is(0)) {
            symbol().init_from(NumericConstant::with_value(0));
            return true;
        }

        if (is_always_positive(arg())) {
            symbol().init_from(NumericConstant::with_value(1));
            return true;
        }

        if (arg().is(Type::Sign)) {
            arg().move_to(symbol());
            return true;
        }

        size_t variable_expressions_size = 0;
        size_t variable_expression_count = 0;
        int sign = 1;
        for (TreeIterator<Product> iterator(&arg()); iterator.is_valid(); iterator.advance()) {
            if (iterator.current()->is(Type::NumericConstant)) {
                sign *= iterator.current()->as<NumericConstant>().value > 0 ? 1 : -1;
            }
            else if (!is_always_positive(*iterator.current())) {
                iterator.current()->copy_to(help_space[variable_expressions_size]);
                variable_expression_count += 1;
                variable_expressions_size += iterator.current()->size();
            }
        }

        if (variable_expression_count == 0) {
            symbol().init_from(NumericConstant::with_value(sign));
            return true;
        }

        Symbol* new_sign = &symbol();
        const size_t neg_size = Neg<None>::Size::get_value();

        if (sign < 0) {
            Neg<None>::init(symbol(), {});
            new_sign += neg_size;
        }

        new_sign->init_from(Sign::builder());

        for (size_t i = 0; i < variable_expression_count - 1; ++i) {
            new_sign->as<Sign>().arg()[i].init_from(Product::builder());
        }

        Symbol::copy_symbol_sequence(&new_sign->as<Sign>().arg() + variable_expression_count - 1,
                                     help_space, variable_expressions_size);

        for (ssize_t i = static_cast<ssize_t>(variable_expression_count) - 2; i >= 0; ++i) {
            new_sign->as<Sign>().arg()[i].as<Product>().seal_arg1();
            new_sign->as<Sign>().arg()[i].as<Product>().seal();
        }

        new_sign->as<Sign>().seal();

        if (sign < 0) {
            (new_sign - neg_size)->size() = new_sign->size() + neg_size;
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
