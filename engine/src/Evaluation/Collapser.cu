#include "Collapser.cuh"

#include "Integrator.cuh"
#include "Symbol/ReverseTreeIterator.cuh"

namespace Sym::Collapser {
    std::vector<Symbol> replace_nth_with_tree(std::vector<Symbol> expression, const size_t n,
                                              const std::vector<Symbol>& tree) {
        if constexpr (Consts::DEBUG) {
            if (!tree[0].is(Type::SubexpressionCandidate)) {
                Util::crash("Invalid first symbol of tree: %s, should be SubexpressionCandidate. "
                            "Whole tree:\n%s\n",
                            type_name(tree[0].type()), tree.data()->to_string().c_str());
            }
        }

        std::vector<Symbol> tree_content;

        if (tree[1].is(Type::Solution)) {
            tree_content = tree[1].as<Solution>().substitute_substitutions();
        }
        else {
            tree_content.resize(tree.size() - 1);
            std::copy(tree.begin() + 1, tree.end(), tree_content.begin());
        }

        expression[n].init_from(ExpanderPlaceholder::with_size(tree_content.size()));

        // This is 1 more than needed, but `compress_to`s implementation expects this
        std::vector<Symbol> new_tree(expression.size() + tree_content.size());
        auto new_tree_iterator =
            SymbolIterator::from_at(*new_tree.data(), 0, new_tree.size()).good();
        expression.data()->compress_to(new_tree_iterator).unwrap();

        std::copy(tree_content.begin(), tree_content.end(),
                  new_tree.begin() + static_cast<int64_t>(n));

        // Resize to what is actually necessary
        new_tree.resize(new_tree.size() - 1);

        return new_tree;
    }

    std::vector<Symbol> collapse_nth(const std::vector<std::vector<Symbol>>& tree, const size_t n) {
        std::vector<Symbol> current_collapse = tree[n];

        for (size_t i = 0; i < current_collapse.size(); ++i) {
            if (!current_collapse[i].is(Type::SubexpressionVacancy)) {
                continue;
            }

            const auto subtree =
                collapse_nth(tree, current_collapse[i].as<SubexpressionVacancy>().solver_idx);

            auto new_collapse = replace_nth_with_tree(current_collapse, i, subtree);
            i += new_collapse.size() - current_collapse.size();
            current_collapse = new_collapse;
        }

        return current_collapse;
    }

    void remove_constants_from_sum(std::vector<Symbol>& expression) {
        if (expression.data()->is_constant()) {
            expression[0].init_from(NumericConstant::with_value(0));
            expression.resize(1);
            return;
        }

        if (!expression.data()->is(Type::Addition)) {
            return;
        }

        auto* const addition = expression.data()->as_ptr<Addition>();

        for (auto* last = addition->last_in_tree(); last >= addition;
             last = (&last->symbol() - 1)->as_ptr<Addition>()) {
            auto& old_arg2 = last->arg2();
            last->seal_arg1();
            old_arg2.move_to(last->arg2());
            last->seal();

            if (last->arg2().is_constant()) {
                last->arg1().move_to(last->symbol());
                continue;
            }

            if (last->arg1().is_constant()) {
                last->arg2().move_to(last->symbol());
            }
        }

        expression.resize(expression.data()->size());
    }

    std::vector<Symbol> collapse(const std::vector<std::vector<Symbol>>& tree) {
        auto collapsed = collapse_nth(tree, 0);
        std::vector<Symbol> reversed(collapsed.size());

        bool success = false;
        while (!success) {
            auto dst_iterator =
                SymbolIterator::from_at(*reversed.data(), 0, reversed.size()).good();

            const auto new_size_res = collapsed.data()->compress_reverse_to(dst_iterator);

            if (new_size_res.is_good()) {
                Symbol::copy_and_reverse_symbol_sequence(*collapsed.data(), *reversed.data(),
                                                         new_size_res.good());
                success = true;
            }
            else {
                reversed.resize(reversed.size() * Integrator::REALLOC_MULTIPLIER);
            }
        }

        std::vector<Symbol> help_space(collapsed.size() * Integrator::HELP_SPACE_MULTIPLIER);
        std::vector<Symbol> collapsed_simplified;

        success = false;
        while (!success) {
            collapsed_simplified = collapsed;
            auto help_space_iterator =
                SymbolIterator::from_at(*help_space.data(), 0, help_space.size()).good();
            success = collapsed_simplified.data()->simplify(help_space_iterator).is_good();

            if (!success) {
                collapsed.resize(collapsed.size() * Integrator::REALLOC_MULTIPLIER);
                help_space.resize(help_space.size() * Integrator::REALLOC_MULTIPLIER);
            }
        }

        collapsed_simplified.resize(collapsed_simplified.data()->size());

        remove_constants_from_sum(collapsed_simplified);

        return collapsed_simplified;
    }
}
