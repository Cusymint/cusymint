#ifndef COLLAPSER_CUH
#define COLLAPSER_CUH

#include <vector>

#include "Symbol/Symbol.cuh"

namespace Sym::Collapser {
    /*
     * @brief Replaces nth symbol in `expression` with `tree`, skipping the first element of
     * `tree` and expanding substitutions if `Solution` is the second symbol in `tree`
     *
     * @param expression Expression to make the replacement in
     * @param n Index of symbol to replace
     * @param tree Expression to make replacement with. Its first symbol is skipped (assumed to
     * be SubexpressionCandidate)
     *
     * @return Copy of `expression` with the replacement
     */
    std::vector<Symbol> replace_nth_with_tree(std::vector<Symbol> expression,
                                                   const size_t n,
                                                   const std::vector<Symbol>& tree);

    /*
     * @brief Collapses a tree of expressions with Solutions with Substitutions and
     * interreferencing SubexpressionCandidates and SubexpressionVacancies to a single
     * expression.
     *
     * @param tree Tree to collapse
     * @param n Index of tree node serving as tree root
     *
     * @return Collapsed tree
     */
    std::vector<Symbol> collapse_nth(const std::vector<std::vector<Symbol>>& tree,
                                          const size_t n);

    /*
     * @brief Removes constant terms from `expression`, when `expression` is of type `Addition`.
     * If `expression` is not sum and is constant, changes `expression` to 0.
     *
     * @param `expression` Expression to remove constants from.
     */
    void remove_constants_from_sum(std::vector<Symbol>& expression);

    /*
     * @brief Collapses a tree of expressions with Solutions with Substitutions and
     * interreferencing SubexpressionCandidates and SubexpressionVacancies to a single
     * expression
     *
     * @param tree Tree to collapse
     *
     * @return Collapsed tree
     */
    std::vector<Symbol> collapse(const std::vector<std::vector<Symbol>>& tree);

}

#endif