#ifndef SCANNER_H
#define SCANNER_H

#include <string>

enum class Token {
    Error = -1,
    Start,
    End,
    Plus,
    Minus,
    Dot,
    Dash,
    Caret, // ok
    Integer,
    Double,
    SymbolicConstant,
    Variable, // ok
    OpenBrace,
    CloseBrace,
    Underscore, // ok
    E,
    Pi,
    Asin,
    Acos,
    Atan,
    Acot, // ok
    Cos,
    Cot,
    Cosh,
    Coth, // ok
    Ln,
    Log, // ok
    Sin,
    Sinh,
    Sqrt, // ok
    Tan,
    Tanh // ok
};

bool isLetter(char c);

class Scanner {
  private:
    std::string text;
    int pos = -1;

    Token read_after_start(Token& state, std::string& read_text);
	Token try_read_inverse_trig(std::string& read_text);
  Token try_read_cosine_cotangent(std::string& read_text);
	Token try_read_log(std::string& read_text);
  Token try_read_sine_sqrt(std::string& read_text);
  Token try_read_tangent(std::string& read_text);
  Token check_if_no_letter_ahead(std::string& read_text, Token return_on_success);

  public:
    Scanner(std::string& text);
    Token scan(std::string& read_text);
};

#endif