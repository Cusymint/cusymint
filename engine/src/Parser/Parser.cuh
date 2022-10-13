#ifndef PARSER_H
#define PARSER_H

#include "../Symbol/Symbol.cuh"
#include "Scanner.cuh"
#include <string>

using SymbolicFunction = std::vector<Sym::Symbol> (*)(const std::vector<Sym::Symbol>&);
using SymbolicOperator = std::vector<Sym::Symbol> (*)(const std::vector<Sym::Symbol>&,
                                                      const std::vector<Sym::Symbol>&);

bool isFunction(Token tok);

// Produkcje:
//
// expr -> term { addop term }					łączny lewostronnie
// term -> factor { mulop factor }				łączny lewostronnie
// factor -> power_arg | power_arg ^ factor		łączny prawostronnie
// power_arg -> num | const | var | ( expr ) | function ( expr )
//
class Parser {
  private:
    Token tok = Token::Start;
    Scanner* scanner;
    std::string read_text;

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
