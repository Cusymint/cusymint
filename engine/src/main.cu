#include <cstdlib>
#include <cstring>

#include <iostream>
#include <optional>
#include <vector>

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "Evaluation/Integrate.cuh"

#include "Symbol/Integral.cuh"
#include "Symbol/Symbol.cuh"

static constexpr size_t BLOCK_SIZE = 1024;
static constexpr size_t BLOCK_COUNT = 32;

std::optional<std::vector<Sym::Symbol>> solve_integral(const std::vector<Sym::Symbol>& integral) {
    Sym::ExpressionArray<> expressions({integral}, Sym::EXPRESSION_MAX_SYMBOL_COUNT,
                                       Sym::MAX_EXPRESSION_COUNT);
    Sym::ExpressionArray<> expressions_swap(Sym::MAX_EXPRESSION_COUNT,
                                            Sym::EXPRESSION_MAX_SYMBOL_COUNT, expressions.size());

    Sym::ExpressionArray<Sym::SubexpressionCandidate> integrals(Sym::MAX_EXPRESSION_COUNT,
                                                                Sym::EXPRESSION_MAX_SYMBOL_COUNT);
    Sym::ExpressionArray<Sym::SubexpressionCandidate> integrals_swap(
        Sym::MAX_EXPRESSION_COUNT, Sym::EXPRESSION_MAX_SYMBOL_COUNT);
    Sym::ExpressionArray<> help_spaces(Sym::MAX_EXPRESSION_COUNT, Sym::EXPRESSION_MAX_SYMBOL_COUNT,
                                       integrals.size());
    Util::DeviceArray<size_t> scan_array_1(Sym::SCAN_ARRAY_SIZE, true);
    Util::DeviceArray<size_t> scan_array_2(Sym::SCAN_ARRAY_SIZE, true);

    // TODO: Przenieść pierwszą całkę z `integral` do `intregrals` i dodać `SubexpressionVacancy`

    for (;;) {
        Sym::simplify<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, help_spaces);
        cudaDeviceSynchronize();

        Sym::check_for_known_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, scan_array_1);
        cudaDeviceSynchronize();

        thrust::inclusive_scan(thrust::device, scan_array_1.begin(), scan_array_1.end(),
                               scan_array_1.data());
        cudaDeviceSynchronize();

        Sym::apply_known_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, expressions, help_spaces,
                                                                scan_array_1);
        cudaDeviceSynchronize();
        std::swap(integrals, integrals_swap);

        Sym::propagate_solved_subexpressions<<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions);
        cudaDeviceSynchronize();

        std::vector<Sym::Symbol> first_expression = expressions.to_vector(0);
        if (first_expression.data()->subexpression_vacancy.is_solved == 1) {
            // TODO: Zwycięstwo, jakoś teraz trzeba zwinąć całe drzewo, zrobić podstawienia i można
            // zwracać wynik
            return std::vector<Sym::Symbol>();
        }

        scan_array_1.zero_mem();
        Sym::find_redundand_expressions<<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions, scan_array_1);
        cudaDeviceSynchronize();

        scan_array_2.zero_mem();
        Sym::find_redundand_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, expressions,
                                                                   scan_array_1, scan_array_2);

        thrust::inclusive_scan(thrust::device, scan_array_1.begin(), scan_array_1.end(),
                               scan_array_1.data());
        thrust::inclusive_scan(thrust::device, scan_array_2.begin(), scan_array_2.end(),
                               scan_array_2.data());
        cudaDeviceSynchronize();

        Sym::remove_expressions_1<<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions, scan_array_1,
                                                               expressions_swap);
        std::swap(expressions, expressions_swap);
        Sym::remove_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, scan_array_2, scan_array_1,
                                                           integrals_swap);
        std::swap(integrals, integrals_swap);
        cudaDeviceSynchronize();

        Sym::check_heuristics_applicability<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, expressions,
                                                                         scan_array_1);
        cudaDeviceSynchronize();
    }

    return std::nullopt;
}

void test_substitutions() {
    std::cout << "Testing manual substitutions" << std::endl;
    std::vector<Sym::Symbol> ixpr = Sym::integral(Sym::var() ^ Sym::num(2));
    std::cout << "Expression 1: " << ixpr.data()->to_string() << std::endl;

    std::vector<Sym::Symbol> ixpr2 = Sym::substitute(ixpr, Sym::cos(Sym::var()));
    std::cout << "Expression 2: " << ixpr2.data()->to_string() << std::endl;

    std::vector<Sym::Symbol> ixpr3 = Sym::substitute(ixpr2, Sym::var() * (Sym::e() ^ Sym::var()));
    std::cout << "Expression 3: " << ixpr3.data()->to_string() << std::endl;
}

void simplify_integrals(Sym::ExpressionArray<Sym::Integral>& integrals,
                        Sym::ExpressionArray<>& help_spaces) {
    std::cout << "Simplifying" << std::endl;

    Sym::simplify<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, help_spaces);
}

std::vector<std::vector<Sym::Symbol>> create_test_integrals() {
    std::cout << "Creating integrals" << std::endl;
    std::vector<std::vector<Sym::Symbol>> integrals{
        Sym::substitute(Sym::integral(Sym::cos(Sym::var())), Sym::pi() * Sym::var()),
        Sym::integral((Sym::sin(Sym::var()) ^ Sym::num(2.0)) +
                      (Sym::num(-8) + (Sym::cos(Sym::var()) ^ Sym::num(2.0)) + Sym::num(4))),
        Sym::integral((Sym::var() + Sym::num(1.0)) +
                      (Sym::pi() + (Sym::e() + Sym::cos(Sym::var())))),
        Sym::integral((Sym::var() ^ Sym::num(2.0)) ^ Sym::e()),
        Sym::integral(Sym::num(2.0) ^ Sym::num(3.0)),
        Sym::integral(Sym::cos(Sym::var()) ^ Sym::num(0.0)),
        Sym::integral(Sym::cos(Sym::sin(Sym::var()))),
        Sym::integral(Sym::e() ^ Sym::var()),
        Sym::integral((Sym::e() ^ Sym::var()) * (Sym::e() ^ Sym::var())),
        Sym::integral(Sym::var() ^ (-Sym::num(5.0))),
        Sym::integral(Sym::var() ^ (Sym::pi() - Sym::num(1.0))),
        Sym::integral(Sym::var() ^ Sym::var()),
        Sym::integral(Sym::pi() + Sym::e() * Sym::num(10.0)),
        Sym::integral((Sym::e() ^ Sym::var()) * (Sym::e() ^ (Sym::e() ^ Sym::var()))),
        Sym::integral(Sym::arccot(Sym::var())),
        Sym::integral(Sym::num(1.0) / ((Sym::var() ^ Sym::num(2.0)) + Sym::num(1.0))),
        Sym::integral(Sym::num(1.0) / (Sym::num(1.0) + (Sym::var() ^ Sym::num(2.0)))),
        Sym::integral(Sym::sinh(Sym::var())),
        Sym::integral(-Sym::num(-5.0))};

    for (const auto& integral : integrals) {
        std::cout << integral.data()->to_string() << std::endl;
    }
    std::cout << std::endl;

    return integrals;
}

void check_and_apply_heuristics(Sym::ExpressionArray<Sym::Integral>& integrals,
                                Sym::ExpressionArray<Sym::Integral>& integrals_swap,
                                Sym::ExpressionArray<>& help_spaces,
                                Util::DeviceArray<size_t>& applicability) {
    std::cout << "Checking heuristics" << std::endl;

    cudaDeviceSynchronize();
    // Sym::check_heuristics_applicability<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, applicability);

    cudaDeviceSynchronize();
    thrust::inclusive_scan(thrust::device, applicability.begin(), applicability.end(),
                           applicability.data());

    cudaDeviceSynchronize();
    Sym::apply_heuristics<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, integrals_swap, help_spaces,
                                                       applicability);
    integrals.resize_from_device(applicability.last());
    integrals_swap.resize(integrals.size());
    help_spaces.resize(integrals.size());

    std::swap(integrals, integrals_swap);

    std::cout << std::endl;
}

void print_applicability(const Util::DeviceArray<size_t>& applicability) {
    const auto h_applicability = applicability.to_vector();

    std::cout << "Applicability:" << std::endl;
    for (size_t i = 0; i < h_applicability.size(); ++i) {
        if (i % Sym::MAX_EXPRESSION_COUNT == 0 && i != 0) {
            std::cout << std::endl;
        }

        std::cout << h_applicability[i] << ", ";
    }
    std::cout << std::endl;
}

void check_and_apply_known_itegrals(Sym::ExpressionArray<Sym::Integral>& integrals,
                                    Sym::ExpressionArray<Sym::Integral>& integrals_swap,
                                    Sym::ExpressionArray<>& help_spaces,
                                    Util::DeviceArray<size_t>& applicability) {
    std::cout << "Checking for known integrals" << std::endl;

    cudaDeviceSynchronize();
    Sym::check_for_known_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, applicability);

    cudaDeviceSynchronize();
    thrust::inclusive_scan(thrust::device, applicability.begin(), applicability.end(),
                           applicability.data());

    cudaDeviceSynchronize();
    Sym::apply_known_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, integrals_swap, help_spaces,
                                                            applicability);
    integrals.resize_from_device(applicability.last());
    integrals_swap.resize(integrals.size());
    help_spaces.resize(integrals.size());

    std::swap(integrals, integrals_swap);

    std::cout << std::endl;
}

void print_results(const Sym::ExpressionArray<Sym::Integral> integrals) {
    const auto h_integrals = integrals.to_vector();

    std::cout << "Results (" << integrals.size() << "):" << std::endl;
    for (size_t int_idx = 0; int_idx < integrals.size(); ++int_idx) {
        std::cout << h_integrals[int_idx].data()->to_string() << std::endl;
    }

    std::cout << std::endl;
}

int main() {
    std::vector<std::vector<Sym::Symbol>> h_integrals = create_test_integrals();

    std::cout << "Allocating and zeroing GPU memory" << std::endl << std::endl;

    Sym::ExpressionArray<Sym::Integral> integrals(h_integrals, Sym::EXPRESSION_MAX_SYMBOL_COUNT,
                                                  Sym::MAX_EXPRESSION_COUNT);
    Sym::ExpressionArray<Sym::Integral> integrals_swap(
        Sym::EXPRESSION_MAX_SYMBOL_COUNT, Sym::MAX_EXPRESSION_COUNT, h_integrals.size());
    Sym::ExpressionArray<> help_spaces(Sym::EXPRESSION_MAX_SYMBOL_COUNT, Sym::MAX_EXPRESSION_COUNT,
                                       h_integrals.size());

    Util::DeviceArray<size_t> applicability(Sym::SCAN_ARRAY_SIZE, true);

    print_results(integrals);

    simplify_integrals(integrals, help_spaces);
    print_results(integrals);

    check_and_apply_heuristics(integrals, integrals_swap, help_spaces, applicability);
    print_results(integrals);
    applicability.zero_mem();

    simplify_integrals(integrals, help_spaces);
    print_results(integrals);

    check_and_apply_known_itegrals(integrals, integrals_swap, help_spaces, applicability);
    print_results(integrals);
    applicability.zero_mem();

    simplify_integrals(integrals, help_spaces);
    print_results(integrals);
}
