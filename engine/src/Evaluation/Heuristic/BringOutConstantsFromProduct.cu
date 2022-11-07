#include "BringOutConstantsFromProduct.cuh"

#include <cuda/std/utility>

#include "Symbol/TreeIterator.cuh"

namespace Sym::Heuristic {
    namespace {
        __device__ cuda::std::pair<size_t, size_t>
        count_constants_and_functions(const Product& tree) {
            size_t constant_count = 0;
            size_t function_count = 0;

            for (ConstTreeIterator<Product> iterator(&tree); iterator.is_valid();
                 iterator.advance()) {
                if (iterator.current()->is_constant()) {
                    constant_count += 1;
                }
                else {
                    function_count += 1;
                }
            }

            return {constant_count, function_count};
        }

        __device__ void create_product_builders(Symbol& dst, const size_t product_count) {
            // Make clear to runtime checks that we can index `dst` arbitrarily
            dst.size() = BUILDER_SIZE;

            for (size_t i = 0; i < product_count; ++i) {
                dst[i].init_from(Product::builder());
            }
        }

        __device__ void seal_product_builders(Symbol& tree, const size_t product_count) {
            for (ssize_t i = static_cast<ssize_t>(product_count) - 1; i >= 0; --i) {
                tree[i].as<Product>().seal_arg1();
                tree[i].as<Product>().seal();
            }
        }

        __device__ void copy_constants_and_functions(Symbol& constants_dst, Symbol& functions_dst,
                                                     const Product& product_tree) {
            Symbol* next_constant_dst = &constants_dst;
            Symbol* next_function_dst = &functions_dst;

            for (ConstTreeIterator<Product> iterator(&product_tree); iterator.is_valid();
                 iterator.advance()) {
                if (iterator.current()->is_constant()) {
                    iterator.current()->copy_to(next_constant_dst);
                    next_constant_dst += iterator.current()->size();
                }
                else {
                    iterator.current()->copy_to(next_function_dst);
                    next_function_dst += iterator.current()->size();
                }
            }
        }
    }

    __device__ CheckResult contains_constants_product(const Integral& integral) {
        if (!integral.integrand()->is(Type::Product)) {
            return CheckResult::empty();
        }

        ConstTreeIterator<Product> iterator(&integral.integrand()->as<Product>());

        // We want to find a constant AND something that is not a constant.
        // If we found only constants, we could in theory return success, but that would mean
        // handling an edge case during application, and we can do without that.
        bool found_constant = false;
        bool found_function = false;

        while (iterator.is_valid()) {
            const bool is_constant = iterator.current()->is_constant();
            found_constant |= is_constant;
            found_function |= !is_constant;

            iterator.advance();

            if (found_constant && found_function) {
                return {1, 1};
            }
        }

        return CheckResult::empty();
    }

    __device__ void bring_out_constants_from_product(
        const SubexpressionCandidate& integral, const ExpressionArray<>::Iterator& integral_dst,
        const ExpressionArray<>::Iterator& expression_dst, Symbol& /*help_space*/) {
        
        const auto& product_tree = integral.arg().as<Integral>().integrand()->as<Product>();
        const auto [constant_count, function_count] = count_constants_and_functions(product_tree);

        auto* expression_candidate = *expression_dst << SubexpressionCandidate::builder();
        expression_candidate->copy_metadata_from(integral);
        expression_candidate->subexpressions_left = 1;

        auto* integral_candidate = *integral_dst << SubexpressionCandidate::builder();
        integral_candidate->subexpressions_left = 0;
        integral_candidate->vacancy_expression_idx = expression_dst.index();

        integral.arg().as<Integral>().copy_without_integrand_to(&integral_candidate->arg());
        auto& integrand_dst = *integral_candidate->arg().as<Integral>().integrand();

        // As many products as constants, because one additional product for the vacancy is needed
        create_product_builders(expression_candidate->arg(), constant_count);

        // One product between every non-constant
        create_product_builders(integrand_dst, function_count - 1);

        expression_candidate->arg()[constant_count].init_from(
            SubexpressionVacancy::for_single_integral());

        copy_constants_and_functions(expression_candidate->arg()[constant_count + 1],
                                     integrand_dst[function_count - 1], product_tree);

        seal_product_builders(expression_candidate->arg(), constant_count);
        seal_product_builders(integrand_dst, function_count - 1);

        integral_candidate->arg().as<Integral>().seal();

        integral_candidate->vacancy_idx = constant_count + 1;
        integral_candidate->seal();

        expression_candidate->seal();
    }
}
