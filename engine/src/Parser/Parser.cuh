#ifndef PARSER_H
#define PARSER_H

#include "../Symbol/Symbol.cuh"
#include "Scanner.cuh"
#include <string>
#include <vector>

using SymbolicFunction = std::vector<Sym::Symbol> (*)(const std::vector<Sym::Symbol>&);
using SymbolicOperator = std::vector<Sym::Symbol> (*)(const std::vector<Sym::Symbol>&,
                                                      const std::vector<Sym::Symbol>&);

std::vector<Sym::Symbol> parse_function(std::string text);

// Production rules:
//
// integral -> expr | int_symbol expr | int_symbol expr 'dx'
// expr -> term { addop term }					        left-associative
// term -> factor { mulop factor }				      left-associative
// factor -> power_arg | power_arg ^ factor		  right-associative
// power_arg -> num | const | var | ( expr ) | log '_' power_arg ( expr ) | function ( expr )
// function -> arcsin | arccos | arctg | arctan | arcctg | arccot | cos | ctg | cot | cosh |
//             ctgh | coth | sin | sinh | sqrt | tg | tan | tgh | tanh | ln
//
// Regardless of the presence of `Integral` token in input string, 
// parser does not wrap expression in `Sym::Integral` symbol.
class Parser {
  private:
    Token tok = Token::Start;
    Scanner* scanner;
    std::string read_text;

    std::vector<Sym::Symbol> integral();
    std::vector<Sym::Symbol> expr();
    std::vector<Sym::Symbol> term();
    std::vector<Sym::Symbol> factor();
    std::vector<Sym::Symbol> power_arg();
    SymbolicFunction function();

    void next_token();
    void throw_error();
    void match_and_get_next_token(Token token);

  public:
    explicit Parser(Scanner* scanner);
    std::vector<Sym::Symbol> parse();
};

#endif
