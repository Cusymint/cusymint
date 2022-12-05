#include "IntegrateByParts.cuh"

#include "Symbol/MetaOperators.cuh"

namespace Sym {
    __host__ __device__ void
    extract_second_factor(const Product& product, const Symbol& first_factor, Symbol& destination) {
        // assume that first_factor and product are sorted
        ConstTreeIterator<Product> product_it(&product);
        ConstTreeIterator<Product> first_factor_it(&first_factor);

        

        while (product_it.is_valid() && first_factor_it.is_valid()) {
            
        }
    }

    __device__ void integrate_by_parts(const SubexpressionCandidate& integral,
                                       const Symbol& first_function_derivative,
                                       const Symbol& first_function,
                                       const ExpressionArray<>::Iterator& integral_dst,
                                       const ExpressionArray<>::Iterator& expression_dst,
                                       Symbol& help_space) {
        const auto& integrand = integral.arg().as<Integral>().integrand()->as<Product>();
    }
}