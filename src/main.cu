#include <cstdlib>
#include <cstring>

#include <iostream>
#include <vector>

#include "symbol.cuh"

int main() {
    std::cout << "Creating an expression" << std::endl;
    std::vector<Sym::Symbol> expression =
        Sym::num(5) / Sym::cnst("f(a)") + Sym::cos(Sym::sin(Sym::tan(Sym::cot(Sym::num(123))))) +
        Sym::num(10) + (Sym::e() ^ (-Sym::pi())) - Sym::cos(Sym::var()) / Sym::sin(Sym::cnst("a"));
    std::cout << "Expression created" << std::endl;

    std::cout << "Printing expression" << std::endl;
    std::cout << expression[0].to_string() << std::endl;
    std::cout << "Expression printed" << std::endl;
}
