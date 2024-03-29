#ifndef PARSER_H
#define PARSER_H

#include <string>
#include <vector>

#include "Symbol/Symbol.cuh"
#include "Scanner.cuh"

namespace Parser {
    using SymbolicFunction = std::vector<Sym::Symbol> (*)(const std::vector<Sym::Symbol>&);
    using SymbolicOperator = std::vector<Sym::Symbol> (*)(const std::vector<Sym::Symbol>&,
                                                          const std::vector<Sym::Symbol>&);

    std::vector<Sym::Symbol> parse_function(const std::string& text);

    // Production rules:
    //
    // integral -> expr | int_symbol expr | int_symbol expr 'dx'
    // expr -> term { addop term }					                      left-associative
    // term -> factor { mulop factor }            	              left-associative; mulop can be empty
    // factor -> '-' factor | power_arg | power_arg ^ factor		  right-associative
    // power_arg -> num | const | var | braced_expr | function_power braced_expr
    // function_power -> function | function ^ factor
    // function -> log '_' power_arg | abs | arcsin | arccos | arctg | arctan | arcctg |
    //             arccot | cos | ctg | cot | cosh | ctgh | coth | sgn | sin | sinh | 
    //             sqrt | tg | tan | tgh | tanh | ln | erf | Si | Ci | Ei | li
    //
    class Parser {
      private:
        Token tok = Token::Start;
        Scanner* scanner;
        std::string read_text;

        std::vector<Sym::Symbol> integral();
        std::vector<Sym::Symbol> expr();
        std::vector<Sym::Symbol> braced_expr();
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

}

#endif
