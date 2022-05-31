#include "parser.hpp"

bool isFunction(Token tok)
{
	return (int)tok >= (int)Token::Asin;
}

Parser::Parser(Scanner *scanner)
{
	this->scanner = scanner;
}

void Parser::next_token()
{
	tok = scanner->scan(read_text);
}

void Parser::throw_error()
{
	throw ("Unexpected token: " + read_text);
}

void Parser::match_and_get_next_token(Token token)
{
	if (tok == token)
		next_token();
	else
		throw_error();
}

std::vector<Sym::Symbol> Parser::expr()
{
	std::vector<Sym::Symbol> sum = term();
	while (tok == Token::Plus || tok == Token::Minus)
	{
		SymbolicOperator op;
		if (tok == Token::Plus)
		{
			op = Sym::operator+;
		}
		else
		{
			op = Sym::operator-;
		}
		next_token();
		sum = op(sum, term());
	}
	return sum;
}

std::vector<Sym::Symbol> Parser::term()
{
	std::vector<Sym::Symbol> product = factor();
	while (tok == Token::Dot || tok == Token::Dash)
	{
		SymbolicOperator op;
		if (tok == Token::Dot)
		{
			op = Sym::operator*;
		}
		else
		{
			op = Sym::operator/;
		}
		next_token();
		product = op(product, factor());
	}
	return product;
}

std::vector<Sym::Symbol> Parser::factor()
{
	std::vector<Sym::Symbol> fact = power_arg();
	while (tok == Token::Caret)
	{
		next_token();
		fact = fact ^ power_arg(); // popraw!
	}
	return fact;
}

std::vector<Sym::Symbol> Parser::power_arg()
{
	std::vector<Sym::Symbol> e;
	switch (tok)
	{
	case Token::Integer:
	case Token::Double:
		next_token();
		return Sym::num(atof(read_text.c_str()));
	case Token::SymbolicConstant:
		next_token();
		return Sym::cnst(read_text.c_str());
	case Token::Variable:
		next_token();
		return Sym::var();
	case Token::E:
		next_token();
		return Sym::e();
	case Token::Pi:
		next_token();
		return Sym::pi();
	case Token::OpenBrace:
		next_token(); // (
		e = expr();
		match_and_get_next_token(Token::CloseBrace); // )
		return e;
	case Token::Minus:
		next_token(); // -
		return -power_arg();
	default:
		if (isFunction(tok))
		{
			SymbolicFunction f = function();
			match_and_get_next_token(Token::OpenBrace); // (
			e = expr();
			match_and_get_next_token(Token::CloseBrace); // )
			return f(e);
		}
		else
		{
			throw_error();
		}
		break;
	}
}

SymbolicFunction Parser::function()
{
	const SymbolicFunction empty = [](const std::vector<Sym::Symbol> &s) { return s; };
	const SymbolicFunction functions[] = {/*arcsin*/empty,/*arccos*/empty,/*arctg*/empty,/*arcctg*/empty,Sym::cos,/*cosh*/empty,/*ctgh*/empty,/*ln*/empty,/*log*/empty,Sym::sin, /*sinh*/empty,/*sqrt*/empty, Sym::tan,/*tanh*/empty};
	Token prev = tok;
	next_token();
	return functions[(int)prev - (int)Token::Asin];
}

std::vector<Sym::Symbol> Parser::parse()
{
	if (tok != Token::Start)
	{
		printf("ERROR: Parser has already processed a string.\n");
		return {};
	}
	next_token();
	try
	{
		std::vector<Sym::Symbol> e = expr();
		match_and_get_next_token(Token::End);
		return e;
	}
	catch (std::string msg)
	{
		printf("ERROR: %s\n", msg.c_str());
		return {};
	}
}