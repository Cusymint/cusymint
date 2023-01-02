#include "Symbol.cuh"

#include "Evaluation/Integrator.cuh"
#include "MetaOperators.cuh"
#include "Symbol/SymbolType.cuh"
#include "Utils/Cuda.cuh"
#include "Utils/Result.cuh"
#include "Utils/StaticStack.cuh"

namespace Sym {
    namespace {
        [[nodiscard]] __host__ __device__ bool is_nonnegative_integer(double value) {
            return value >= 0 && floor(value) == value;
        }
    }

    [[nodiscard]] __host__ __device__ bool Symbol::is(const double number) const {
        return is(Type::NumericConstant) && as<NumericConstant>().value == number;
    }

    [[nodiscard]] __host__ __device__ bool Symbol::is_negation() const {
        return Mul<Integer<-1>, Any>::match(*this) || Mul<Any, Integer<-1>>::match(*this);
    }

    [[nodiscard]] __host__ __device__ const Symbol& Symbol::negation_arg() const {
        if (Mul<Integer<-1>, Any>::match(*this)) {
            return as<Product>().arg2();
        }

        if constexpr (Consts::DEBUG) {
            if (!Mul<Any, Integer<-1>>::match(*this)) {
                Util::crash(
                    "Trying to get negated subexpression of something that is not a negation");
            }
        }

        return as<Product>().arg1();
    }

    [[nodiscard]] __host__ __device__ bool Symbol::is_integer() const {
        return is(Type::NumericConstant) &&
               as<NumericConstant>().value == floor(as<NumericConstant>().value);
    }

    __host__ __device__ void Symbol::copy_symbol_sequence(Symbol* const destination,
                                                          const Symbol* const source,
                                                          size_t symbol_count) {
        Util::copy_mem(destination, source, symbol_count * sizeof(Symbol));
    }

    __host__ __device__ void Symbol::move_symbol_sequence(Symbol* const destination,
                                                          Symbol* const source,
                                                          size_t symbol_count) {
        Util::move_mem(destination, source, symbol_count * sizeof(Symbol));
    }

    __host__ __device__ void Symbol::copy_and_reverse_symbol_sequence(Symbol& destination,
                                                                      const Symbol& source,
                                                                      const size_t symbol_count) {
        for (size_t i = 0; i < symbol_count; ++i) {
            (&source)[symbol_count - i - 1].copy_single_to(*destination.at_unchecked(i));
        }
    }

    __host__ __device__ bool Symbol::compare_symbol_sequences(const Symbol* const sequence1,
                                                              const Symbol* const sequence2,
                                                              size_t symbol_count) {
        // Cannot simply use Util::compare_mem because padding can differ
        for (size_t i = 0; i < symbol_count; ++i) {
            if (sequence1[i] != sequence2[i]) {
                return false;
            }
        }

        return true;
    }

    __host__ __device__ void Symbol::swap_symbols(Symbol* const symbol1, Symbol* const symbol2) {
        Util::swap_mem(symbol1, symbol2, sizeof(Symbol));
    }

    __host__ __device__ void Symbol::reverse_symbol_sequence(Symbol* const sequence,
                                                             size_t symbol_count) {
        for (size_t i = 0; i < symbol_count / 2; ++i) {
            swap_symbols(sequence + i, sequence + symbol_count - i - 1);
        }
    }

    __host__ __device__ void Symbol::copy_single_to(Symbol& destination) const {
        Util::copy_mem(&destination, this, sizeof(Symbol));
    }

    __host__ __device__ void Symbol::copy_to(Symbol& destination) const {
        copy_symbol_sequence(&destination, this, size());
    }

    __host__ __device__ void Symbol::move_to(Symbol& destination) {
        move_symbol_sequence(&destination, this, size());
    }

    __host__ __device__ bool Symbol::is_constant() const {
        for (size_t i = 0; i < size(); ++i) {
            if (this[i].is(Type::Variable)) {
                return false;
            }
        }

        return true;
    }

    __host__ __device__ ssize_t Symbol::first_var_occurence() const {
        for (ssize_t i = 0; i < size(); ++i) {
            if (this[i].is(Type::Variable)) {
                return i;
            }
        }

        return -1;
    }

    __host__ __device__ bool Symbol::is_function_of(Symbol* const help_space,
                                                    const Symbol* const* const expressions,
                                                    const size_t expression_count) const {
        bool* const results = reinterpret_cast<bool*>(help_space);

        for (auto i = static_cast<ssize_t>(size()) - 1; i >= 0; --i) {
            results[i] =
                VIRTUAL_CALL(*at(i), is_function_of, expressions, expression_count, results + i);
        }
        return results[0];
    }

    __host__ __device__ void Symbol::seal_whole(Symbol& expr, const size_t size) {
        expr.size() = BUILDER_SIZE;

        for (size_t i = size; i > 0; --i) {
            expr[i - 1].size() = BUILDER_SIZE;
            VIRTUAL_CALL(expr[i - 1], seal_whole);
        }
    }

    __host__ __device__ Util::SimpleResult<size_t>
    Symbol::compress_reverse_to(SymbolIterator destination) {
        const size_t original_destination_idx = destination.index();
        mark_to_be_copied_and_propagate_additional_size(&destination.current());

        for (ssize_t i = static_cast<ssize_t>(size()) - 1; i >= 0; --i) {
            if (!at(i)->to_be_copied()) {
                continue;
            }

            at(i)->to_be_copied() = false;

            const size_t new_size = VIRTUAL_CALL(*at(i), compression_size);
            if (!destination.can_offset_by(new_size)) {
                return Util::SimpleResult<size_t>::make_error();
            }

            VIRTUAL_CALL(*at(i), compress_reverse_to, &destination.current());
            TRY_PASS(size_t, destination += new_size);
        }

        // Corrects the size of copied `this` (if additional demanded)
        Symbol* const last_copied_symbol = &destination.current() - 1;
        last_copied_symbol->size() += last_copied_symbol->additional_required_size();
        last_copied_symbol->additional_required_size() = 0;

        return Util::SimpleResult<size_t>::make_good(destination.index() -
                                                     original_destination_idx);
    }

    __host__ __device__ void
    Symbol::mark_to_be_copied_and_propagate_additional_size(Symbol* const help_space) {
        Util::StaticStack<Symbol*> stack(reinterpret_cast<Symbol**>(help_space));

        stack.push(this);

        while (!stack.empty()) {
            Symbol* const sym = stack.pop();

            sym->to_be_copied() = true;
            VIRTUAL_CALL(*sym, put_children_on_stack_and_propagate_additional_size, stack);
        }
    }

    __host__ __device__ Util::SimpleResult<size_t>
    Symbol::compress_to(SymbolIterator& destination) {
        const size_t new_size = TRY(compress_reverse_to(destination));
        reverse_symbol_sequence(&destination.current(), new_size);
        return Util::SimpleResult<size_t>::make_good(new_size);
    }

    __host__ __device__ Util::BinaryResult Symbol::simplify(SymbolIterator& help_space) {
        const size_t this_capacity = help_space.capacity() / Integrator::HELP_SPACE_MULTIPLIER;

        bool success = false;
        while (!success) {
            success = true;

            for (ssize_t i = static_cast<ssize_t>(size()) - 1; i >= 0; --i) {
                success = at(i)->simplify_in_place(help_space) && success;
            }

            const size_t new_size = TRY_PASS(Util::Empty, compress_reverse_to(help_space));

            if (this_capacity < new_size) {
                return Util::BinaryResult::make_error();
            }

            copy_and_reverse_symbol_sequence(*this, *help_space, new_size);
        }

        return Util::BinaryResult::make_good();
    }

    __host__ __device__ bool Symbol::simplify_in_place(SymbolIterator& help_space) {
        return VIRTUAL_CALL(*this, simplify_in_place, &help_space.current());
    }

    void Symbol::substitute_variable_with(const Symbol symbol) {
        for (size_t i = 0; i < size(); ++i) {
            if (this[i].is(Type::Variable)) {
                this[i] = symbol;
            }
        }
    }

    void Symbol::substitute_variable_with_nth_substitution_name(const size_t n) {
        std::string substitution_name = Substitution::nth_substitution_name(n);

        Symbol substitute{};
        substitute.init_from(UnknownConstant::create(substitution_name.c_str()));
        substitute_variable_with(substitute);
    }

    __host__ __device__ bool Symbol::are_expressions_equal(const Symbol& expr1,
                                                           const Symbol& expr2) {
        if (expr1.size() != expr2.size()) {
            return false;
        }

        return compare_symbol_sequences(&expr1, &expr2, expr1.size());
    }

    __host__ __device__ Util::Order
    Symbol::compare_expressions(const Symbol& expr1, const Symbol& expr2, Symbol& help_space) {
        Util::StaticStack<const Symbol*> expr1_stack(reinterpret_cast<const Symbol**>(&help_space));
        Util::StaticStack<const Symbol*> expr2_stack(reinterpret_cast<const Symbol**>(
            expr1.size() + &help_space)); // expr1.size() should not be smaller than the actual size

        expr1_stack.push(&expr1);
        expr2_stack.push(&expr2);

        while (!expr1_stack.empty() && !expr2_stack.empty()) {
            const Symbol* const expr1_sym = expr1_stack.pop();
            const Symbol* const expr2_sym = expr2_stack.pop();

            if (expr1_sym->type_ordinal() < expr2_sym->type_ordinal()) {
                return Util::Order::Less;
            }

            if (expr1_sym->type_ordinal() > expr2_sym->type_ordinal()) {
                return Util::Order::Greater;
            }

            const auto order = VIRTUAL_CALL(*expr1_sym, compare_to, *expr2_sym);
            if (order != Util::Order::Equal) {
                return order;
            }

            VIRTUAL_CALL(*expr1_sym, push_children_onto_stack, expr1_stack);
            VIRTUAL_CALL(*expr2_sym, push_children_onto_stack, expr2_stack);
        }

        return Util::Order::Equal;
    }

    std::string Symbol::to_string() const { return VIRTUAL_CALL(*this, to_string); }

    std::string Symbol::to_tex() const { return VIRTUAL_CALL(*this, to_tex); }

    __host__ __device__ Util::OptionalNumber<ssize_t>
    Symbol::is_polynomial(Symbol* const help_space) const {
        auto* ranks = reinterpret_cast<Util::OptionalNumber<ssize_t>*>(help_space);
        for (ssize_t i = static_cast<ssize_t>(size()) - 1; i >= 0; --i) {
            const Symbol* const current = at(i);
            switch (current->type()) {
            case Type::Addition: {
                const auto& addition = current->as<Addition>();
                const auto& rank1 = ranks[i + 1];
                const auto& rank2 = ranks[i + addition.second_arg_offset];
                ranks[i] = Util::max(rank1, rank2); // TODO
            } break;
            case Type::NumericConstant:
                ranks[i] = 0;
                break;
            case Type::Power: {
                const auto& power = current->as<Power>();
                if (power.arg1().is(Type::Variable) && power.arg2().is(Type::NumericConstant) &&
                    is_nonnegative_integer(power.arg2().as<NumericConstant>().value)) {
                    ranks[i] = static_cast<ssize_t>(power.arg2().as<NumericConstant>().value);
                }
                else {
                    ranks[i] = Util::empty_num;
                }
            } break;
            case Type::Product: {
                const auto& product = current->as<Product>();
                if (product.arg1().is(Type::Addition) || product.arg2().is(Type::Addition)) {
                    ranks[i] = Util::empty_num;
                    break;
                }
                const auto& rank1 = ranks[i + 1];
                const auto& rank2 = ranks[i + product.second_arg_offset];
                ranks[i] = rank1 + rank2;
            } break;
            case Type::Variable:
                ranks[i] = 1;
                break;
            default:
                ranks[i] = Util::empty_num;
            }
        }
        return ranks[0];
    }

    __host__ __device__ Util::OptionalNumber<double>
    Symbol::get_monomial_coefficient(Symbol* const help_space) const {
        auto* coefficients = reinterpret_cast<Util::OptionalNumber<double>*>(help_space);
        for (ssize_t i = static_cast<ssize_t>(size()) - 1; i >= 0; --i) {
            const Symbol* const current = at(i);
            switch (current->type()) {
            case Type::NumericConstant:
                coefficients[i] = current->as<NumericConstant>().value;
                break;
            case Type::Power:
                coefficients[i] = 1;
                break;
            case Type::Product: {
                const auto& product = current->as<Product>();
                coefficients[i] = coefficients[i + 1] * coefficients[i + product.second_arg_offset];
            } break;
            case Type::Variable:
                coefficients[i] = 1;
                break;
            default:
                coefficients[i] = Util::empty_num;
            }
        }
        return coefficients[0];
    }

    [[nodiscard]] __host__ __device__ Util::SimpleResult<size_t>
    Symbol::derivative_to(SymbolIterator& destination) {
        SymbolIterator current_dst = destination;
        for (auto i = static_cast<ssize_t>(size() - 1); i >= 0; --i) {
            const ssize_t predicted_offset = VIRTUAL_CALL(*at(i), derivative_size, *current_dst);

            if (predicted_offset > 0 && !destination.can_offset_by(predicted_offset)) {
                return Util::SimpleResult<size_t>::make_error();
            }

            const ssize_t offset = VIRTUAL_CALL(*at(i), insert_reversed_derivative_at, *current_dst);
            TRY_PASS(size_t, current_dst += offset);
        }
        const size_t symbols_inserted = current_dst.index() - destination.index();
        reverse_symbol_sequence(&destination.current(), symbols_inserted);
        return Util::SimpleResult<size_t>::make_good(symbols_inserted);
    }

    __host__ __device__ bool operator==(const Symbol& sym1, const Symbol& sym2) {
        return VIRTUAL_CALL(sym1, are_equal, &sym2);
    }

    __host__ __device__ bool operator!=(const Symbol& sym1, const Symbol& sym2) {
        return !(sym1 == sym2);
    }
};
