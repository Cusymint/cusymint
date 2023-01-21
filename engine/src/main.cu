#include <cstddef>
#include <cstdlib>

#include <fmt/format.h>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <fmt/core.h>

#include "Evaluation/Integrator.cuh"
#include "Evaluation/StaticFunctions.cuh"

#include "Performance.cuh"
#include "Symbol/Constants.cuh"
#include "Symbol/ExpressionArray.cuh"
#include "Symbol/Integral.cuh"
#include "Symbol/Symbol.cuh"
#include "Symbol/Variable.cuh"

#include "Parser/Parser.cuh"

#include "Utils/CompileConstants.cuh"

/*
 * @brief Creates a `std::string` representing expression of type `e^x * e^e^x * ... * e^e^...^e^x`,
 * which is made of `n` factors.
 *
 * @param `n` - number of factors in created expression.
 *
 * @return Created string. If `n==0`, function returns `"1"`.
 */
std::string e_tower(size_t n) {
    if (n == 0) {
        return "1";
    }
    std::string res = "e^x";
    for (int i = 2; i <= n; ++i) {
        res += "*";
        for (int j = 1; j <= i; ++j) {
            res += "e^";
        }
        res += "x";
    }
    return res;
}

std::vector<std::string> generate_e_towers(size_t end) {
    std::vector<std::string> result(end + 1);
    result[0] = "1";
    if (end > 0) {
        result[1] = "e^x";
    }
    for (size_t i = 2; i <= end; ++i) {
        std::string e_power = "*";
        for (size_t j = 1; j <= i; ++j) {
            e_power += "e^";
        }
        e_power += "x";
        result[i] = result[i - 1] + e_power;
    }
    return result;
}

std::vector<std::string> generate_sin_towers(size_t end) {
    std::vector<std::string> result(end + 1);
    result[0] = "cos(x)";
    for (size_t i = 1; i <= end; ++i) {
        std::string trig = "*cos(";
        for (size_t j = 0; j < i; ++j) {
            trig += "sin(";
        }
        trig += "x";
        for (size_t j = 0; j <= i; ++j) {
            trig += ")";
        }
        result[i] = result[i - 1] + trig;
    }
    return result;
}

std::vector<std::string> generate_geometric_sums(size_t start, size_t end, size_t step = 1) {
    std::vector<std::string> result;
    for (size_t i = start; i <= end; i += step) {
        result.push_back(fmt::format("(x^{}-1)/(x-1)", i));
    }
    return result;
}

std::vector<std::string> generate_random_polynomials(size_t min_rank, size_t max_rank, size_t count,
                                                     size_t seed) {
    constexpr size_t MAX_COEFF = 50;
    std::vector<std::string> result(count);
    srand(seed);
    for (size_t i = 0; i < count; ++i) {
        size_t rank = (rand() % (max_rank - min_rank + 1)) + min_rank;
        result[i] = fmt::format("{}", rand() % (MAX_COEFF + 1));
        for (size_t j = 1; j <= rank; ++j) {
            result[i] += fmt::format("{}{}*x^{}", (rand() % 2 == 0) ? "+" : "-",
                                     rand() % (MAX_COEFF + 1), j);
        }
    }
    return result;
}

std::vector<std::string> generate_random_trig_polynomials(size_t start, size_t end, size_t step,
                                                          size_t seed) {
    constexpr size_t MAX_COEFF = 50;
    std::vector<std::string> result;
    srand(seed);
    for (size_t rank = start; rank <= end; rank += step) {
        std::string str = fmt::format("{}", rand() % (MAX_COEFF + 1));
        for (size_t j = 1; j <= rank; ++j) {
            str += fmt::format("{}{}*sin(x)^{}*cos(x)", (rand() % 2 == 0) ? "+" : "-",
                               rand() % (MAX_COEFF + 1), j);
        }
        result.push_back(str);
    }
    return result;
}

int main() {
    if constexpr (Consts::DEBUG) {
        fmt::print("Running in debug mode\n");
    }

    Sym::Static::init_functions();

    fmt::print("First two integrals are for warmup!\n");

    std::vector<std::string> integrals = {
        "x+sin(x)", "e^x",
        // "x^4-5*x+3",
        // "(1-x^3)^2",
        // "x^3*t^2",
        // "x^(2/3)",
        // "1/x^(2/3)",
        // "(x^3-3*x^2+1)/x^5",
        // "(x^(2/3)-sqrt(x))/x^(1/3)",
        // "2*e^x+3*5^x",
        // "x*cos(x)",
        // "x*e^x",
        // "x^2*e^x",
        // "x*e^(-x)",
        // "x^2*cos(x)",
        // "sqrt(x)*ln(x)",
        // "ln(x)/x^4",
        // "(2*x-1)^20",
        // "cos(3*x-1)",
        // "e^x/(3+e^x)",
        // "tan(x)",
        // "tan(x)/cos(x)^2",
        // "cos(x)*e^sin(x)",
        // "cos(x)/sqrt(1+sin(x))",
        // "6^(1-x)",
        // "(sqrt(x)-2)^2/x^2",
        // "1/(sin(x)^2*cos(x)^2)",
        // "x*arctan(x)",
        // "abs(x)",
        // "(abs(1-x^2)+1+x^2)/2",
        // "tan(x)^2",
        // "(2*x-3)^10",
        // "sin(x)^5*cos(x)",
        // "ln(x)",
        // "x*cos(x)",
        // "x^2*e^(1-x)",
        // "sin(x)^5",
        // "sin(x)^4*cos(x)^3",
        // "e^x*(x+1)*ln(x)",
        // "(1-x^200)/(1-x)",
        // "(1-sin(x)^20)/(1-sin(x))*cos(x)",
        // e_tower(70),
    };

    // add huge sum
    // integrals.push_back(fmt::format("{}", fmt::join(integrals, "+")));

    // fmt::print("{}", fmt::join(generate_sin_towers(5), ","));

    // add e towers
    // const auto e_towers = generate_e_towers(10);
    // integrals.insert(integrals.end(), e_towers.begin(), e_towers.end());

    // add sin towers
    // const auto sin_towers = generate_sin_towers(8);
    // integrals.insert(integrals.end(), sin_towers.begin(), sin_towers.end());

    // add geometric sums
    const auto geo_sums = generate_geometric_sums(0, 200, 5);
    integrals.insert(integrals.end(), geo_sums.begin(), geo_sums.end());

    // add random polynomials
    const auto polynomials = generate_random_trig_polynomials(0, 5, 100, 2137);
    //integrals.insert(integrals.end(), polynomials.begin(), polynomials.end());

    // Performance::test_performance(integrals, Performance::print_csv_results);
    // Performance::test_memory_occupance(integrals);
    Performance::test_history_cpu_memory_occupance(integrals);
    // const auto integral = Sym::integral(Parser::parse_function(integrals.back()));

    // fmt::print("Trying to solve an integral: {}\n", integral.data()->to_tex());

    // Sym::Integrator integrator;
    // Sym::ComputationHistory history;
    // const auto solution =
    // integrator.solve_integral_with_history(integral, history);

    // if (solution.has_value()) {
    //     fmt::print("Success! Solution:\n{} + C\n", solution.value().data()->to_tex());

    //     fmt::print("\nComputation steps:\n\n");

    //     for (const auto& step: history.get_tex_history()) {
    //         fmt::print("  {}\\\\\n\n", step);
    //     }
    // }
    // else {
    //     fmt::print("No solution found\n");
    // }
}
