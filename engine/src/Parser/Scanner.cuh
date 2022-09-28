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
    Caret,
    Integer,
    Double,
    SymbolicConstant,
    Variable,
    OpenBrace,
    CloseBrace,
    Underscore,
    E,
    Pi,
    Asin,
    Acos,
    Atan,
    Acot,
    Cos,
    Cot,
    Cosh,
    Coth,
    Ln,
    Log,
    Sin,
    Sinh,
    Sqrt,
    Tan,
    Tanh
};

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