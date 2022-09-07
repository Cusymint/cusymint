#ifndef PARSER_H
#define PARSER_H

#include "Scanner.cuh"
#include "../Symbol/Symbol.cuh"
#include <string>

typedef std::vector<Sym::Symbol> (*SymbolicFunction)(const std::vector<Sym::Symbol>&); 
typedef std::vector<Sym::Symbol> (*SymbolicOperator)(const std::vector<Sym::Symbol>&, const std::vector<Sym::Symbol>&); 

bool isFunction(Token tok);

// Produkcje:
//
// expr -> term { addop term }					��czny lewostronnie
// term -> factor { mulop factor }				��czny lewostronnie
// factor -> power_arg | power_arg ^ factor		��czny prawostronnie
// power_arg -> num | const | var | ( expr ) | function ( expr )
//
class Parser
{
private:
	Token tok = Token::Start;
	Scanner *scanner;
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
	Parser(Scanner *scanner);
	std::vector<Sym::Symbol> parse();
};

#endif

