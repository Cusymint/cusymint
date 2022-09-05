#ifndef SCANNER_H
#define SCANNER_H

#include <string>

enum class Token
{
	Error = -1, Start, End, 
	Plus, Minus, Dot, Dash, Caret, // ok
	Integer, Double, SymbolicConstant, Variable, // ok
	OpenBrace, CloseBrace, Underscore, // ok
    E, Pi,
	Asin, Acos, Atan, Acot, // ok
	Cos, Cot, Cosh, Coth, // ok
	Ln, Log, // ok
	Sin, Sinh, Sqrt, // ok
	Tan, Tanh // ok
};

bool isLetter(char c);

class Scanner
{
private:
	std::string text;
	int pos = -1;
public:
	Scanner(std::string text);
	Token scan(std::string &read_text);
};


#endif