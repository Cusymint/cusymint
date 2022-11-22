#include "Power.cuh"

#include "MetaOperators.cuh"
#include "Symbol.cuh"
#include "Symbol/Constants.cuh"
#include "Symbol/ExpanderPlaceholder.cuh"
#include "Symbol/SymbolType.cuh"
#include "Symbol/TreeIterator.cuh"
#include <fmt/core.h>

namespace {
    __host__ __device__ inline bool is_symbol_inverse_logarithm_of(const Sym::Symbol& symbol,
                                                                   const Sym::Symbol& expression) {
        return Sym::PatternPair<Sym::Inv<Sym::Ln<Sym::Same>>, Sym::Same>::match_pair(symbol,
                                                                                     expression);
    }
}

namespace Sym {
    DEFINE_TWO_ARGUMENT_OP_FUNCTIONS(Power)
    DEFINE_SIMPLE_TWO_ARGUMENT_OP_ARE_EQUAL(Power)
    DEFINE_IDENTICAL_COMPARE_TO(Power)
    DEFINE_TWO_ARGUMENT_OP_COMPRESS_REVERSE_TO(Power)

    DEFINE_SIMPLIFY_IN_PLACE(Power) {
        if (arg2().is(Type::NumericConstant) && arg2().as<NumericConstant>().value == 0) {
            symbol()->init_from(NumericConstant::with_value(1));
            return true;
        }

        if (arg2().is(Type::NumericConstant) && arg2().as<NumericConstant>().value == 1) {
            arg1().copy_to(help_space);
            help_space->copy_to(symbol());
            return true;
        }

        if (arg1().is(Type::NumericConstant) && arg2().is(Type::NumericConstant)) {
            double value1 = arg1().as<NumericConstant>().value;
            double value2 = arg2().as<NumericConstant>().value;
            symbol()->init_from(NumericConstant::with_value(pow(value1, value2)));
            return true;
        }

        // (a^b)^c -> a^(b*c)
        if (arg1().is(Type::Power)) {
            Symbol::from(this)->copy_to(help_space);
            Power* const this_copy = &help_space->power;

            *this = Power::create();
            this_copy->arg1().power.arg1().copy_to(&arg1());
            seal_arg1();

            Product* const product = &arg2() << Product::create();
            this_copy->arg1().power.arg2().copy_to(&product->arg1());
            product->seal_arg1();
            this_copy->arg2().copy_to(&product->arg2());
            product->seal();

            seal();

            return false; // b and c may be simplified
        }

        // a^(1/ln(a))=e
        if (is_symbol_inverse_logarithm_of(arg2(), arg1())) {
            symbol()->init_from(KnownConstant::with_value(KnownConstantValue::E));
            return true;
        }

        // e^(ln(b))=b
        if (arg2().is(Type::Logarithm) && arg1().is(Type::KnownConstant) &&
            arg1().as<KnownConstant>().value == KnownConstantValue::E) {
            arg2().as<Logarithm>().arg().copy_to(symbol());
            return true;
        }

        // a^(...*1/ln(a)*...)=e^(...), e^(...*ln(b)*...)=b^(...)
        if (arg2().is(Type::Product)) {
            TreeIterator<Product> iterator(arg2().as_ptr<Product>());
            Symbol* base = &arg1();
            bool base_changed = false;
            while (iterator.is_valid()) {
                if (is_symbol_inverse_logarithm_of(*iterator.current(), *base)) {
                    base->init_from(KnownConstant::with_value(KnownConstantValue::E));
                    iterator.current()->init_from(NumericConstant::with_value(1));
                    base_changed = true;
                }
                if (iterator.current()->is(Type::Logarithm) && base->is(Type::KnownConstant) &&
                    base->as<KnownConstant>().value == KnownConstantValue::E) {
                    iterator.current()->as<Logarithm>().arg().copy_to(help_space);
                    base = help_space;
                    iterator.current()->init_from(NumericConstant::with_value(1));
                    base_changed = true;
                }
                iterator.advance();
            }
            if (base == help_space) {
                arg1().init_from(ExpanderPlaceholder::with_size(base->size()));
                Symbol* const compressed_reversed = base + base->size();
                const auto compressed_size = symbol()->compress_reverse_to(compressed_reversed);
                Symbol::copy_and_reverse_symbol_sequence(symbol(), compressed_reversed,
                                                         compressed_size);
                base->copy_to(&arg1());
            }
            // if power base was changed, there may be remaining ones to simplify
            if (base_changed) {
                arg2().as<Product>().eliminate_ones();
            }
            return true;
        }
        return true;
    }

    DEFINE_IS_FUNCTION_OF(Power) {
        for (size_t i = 0; i < expression_count; ++i) {
            if (!expressions[i]->is(Type::Power)) {
                continue;
            }

            const auto& power_expression = expressions[i]->as<Power>();

            // TODO: In the future, this should look for correspondences in the product tree of
            // arg1(). For example, if `this` is `e^(pi*x*2*sin(x))`, then this function should
            // return true when `power_expression` is `e^(sin(x)*x)`.
            if (Symbol::are_expressions_equal(arg1(), power_expression.arg1()) &&
                Symbol::are_expressions_equal(arg2(), power_expression.arg2())) {
                return true;
            }
        }

        return arg1().is_function_of(expressions, expression_count) &&
               arg2().is_function_of(expressions, expression_count);
    }

    DEFINE_INSERT_REVERSED_DERIVATIVE_AT(Power) {
        const size_t d_arg1_size = (destination - 1)->size();
        Symbol* const rev_arg2 = destination - 1 - d_arg1_size;
        if ((destination - 1)->is(0)) { // arg1() is constant: exponential function or constant
            if (rev_arg2->is(0)) {      // arg2() is constant: constant
                return -1;
            }
            // (expr') c ln (expr) c ^ * *
            return Prod<Pow<Copy, Copy>, Ln<Copy>, None>::init_reverse(*(destination - 1),
                                                                       {arg1(), arg2(), arg1()}) -
                   1;
        }
        if (rev_arg2->is(0)) { // arg2() is constant: monomial
            // (expr') -1 c + (expr) ^ c * *

            Symbol::move_symbol_sequence(rev_arg2, rev_arg2 + 1,
                                         d_arg1_size); // move derivative of arg1() one index back
            return Prod<Copy, Pow<Copy, Add<Copy, Integer<-1>>>, None>::init_reverse(
                       *(destination - 1), {arg2(), arg1(), arg2()}) -
                   1;
        }
        // General case:
        // (expr2') (expr1) ln * (expr1') (expr1) inv (expr2) * * + (expr2) (expr1) ^ *
        Symbol* const new_d_arg1_dst = rev_arg2 + 3 + arg1().size();
        Symbol::move_symbol_sequence(new_d_arg1_dst, rev_arg2 + 1,
                                     d_arg1_size); // copy (expr1')
        return Mul<Pow<Copy, Copy>, Add<Prod<Copy, Inv<Copy>, Skip>, Mul<Ln<Copy>, None>>>::
                   init_reverse(*(rev_arg2 + 1),
                                {arg1(), arg2(), arg2(), arg1(), d_arg1_size, arg1()}) -
               d_arg1_size;
    }

    std::string Power::to_string() const {
        return fmt::format("{}^{}", arg1().to_string(), arg2().to_string());
    }

    std::string Power::to_tex() const {
        if (arg2().is(Type::NumericConstant) && arg2().numeric_constant.value == 0.5) {
            return fmt::format("\\sqrt{{ {} }}", arg1().to_tex());
        }
        if (arg1().is(Type::Addition) || arg1().is(Type::Product) || arg1().is(Type::Negation) ||
            arg1().is(Type::Reciprocal) || arg1().is(Type::Power)) {
            return fmt::format(R"(\left({}\right)^{{ {} }})", arg1().to_tex(), arg2().to_tex());
        }
        return fmt::format("{}^{{ {} }}", arg1().to_tex(), arg2().to_tex());
    }

    std::vector<Symbol> operator^(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs) {
        std::vector<Symbol> res(lhs.size() + rhs.size() + 1);
        Power::create(lhs.data(), rhs.data(), res.data());
        return res;
    }

    std::vector<Symbol> sqrt(const std::vector<Symbol>& arg) { return arg ^ Sym::num(0.5); }
}
