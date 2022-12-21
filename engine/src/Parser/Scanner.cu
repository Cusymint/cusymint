#include "Scanner.cuh"
#include <cctype>
#include <string>

namespace Parser {
    Scanner::Scanner(const std::string& text) : text(text) {}

    bool isFunction(Token tok) { return tok >= Token::Abs; }

    void Scanner::read_letter_sequence(std::string& read_text) {
        while (std::isalpha(text[pos])) {
            read_text += text[pos++];
        }
    }

    Token Scanner::try_read_abs_inverse_trig(std::string& read_text) {
        read_text += 'a';
        if (text.substr(pos + 1, 5) == "rcsin") {
            read_text += "rcsin";
            pos += 5;
            return Token::Asin;
        }
        if (text.substr(pos + 1, 5) == "rccos") {
            read_text += "rccos";
            pos += 5;
            return Token::Acos;
        }
        if (text.substr(pos + 1, 4) == "rctg") {
            read_text += "rctg";
            pos += 4;
            return Token::Atan;
        }
        if (text.substr(pos + 1, 5) == "rcctg") {
            read_text += "rcctg";
            pos += 5;
            return Token::Acot;
        }
        if (text.substr(pos + 1, 5) == "rctan") {
            read_text += "rctan";
            pos += 5;
            return Token::Atan;
        }
        if (text.substr(pos + 1, 5) == "rccot") {
            read_text += "rccot";
            pos += 5;
            return Token::Acot;
        }
        if (text.substr(pos + 1, 2) == "bs") {
            read_text += "bs";
            pos += 2;
            return Token::Abs;
        }
        return Token::SymbolicConstant;
    }

    Token Scanner::try_read_cosine_cotangent(std::string& read_text) {
        read_text += 'c';
        if (text.substr(pos + 1, 3) == "osh") {
            read_text += "osh";
            pos += 3;
            return Token::Cosh;
        }
        if (text.substr(pos + 1, 3) == "tgh") {
            read_text += "tgh";
            pos += 3;
            return Token::Coth;
        }
        if (text.substr(pos + 1, 3) == "oth") {
            read_text += "oth";
            pos += 3;
            return Token::Coth;
        }
        if (text.substr(pos + 1, 2) == "os") {
            read_text += "os";
            pos += 2;
            return Token::Cos;
        }
        if (text.substr(pos + 1, 2) == "tg") {
            read_text += "tg";
            pos += 2;
            return Token::Cot;
        }
        if (text.substr(pos + 1, 2) == "ot") {
            read_text += "ot";
            pos += 2;
            return Token::Cot;
        }
        return Token::SymbolicConstant;
    }

    Token Scanner::try_read_log(std::string& read_text) {
        read_text += 'l';
        if (text.substr(pos + 1, 2) == "og") {
            read_text += "og";
            pos += 2;
            return Token::Log;
        }
        if (text.substr(pos + 1, 1) == "n") {
            read_text += "n";
            ++pos;
            return Token::Ln;
        }
        return Token::SymbolicConstant;
    }

    Token Scanner::try_read_sign_sine_sqrt(std::string& read_text) {
        read_text += 's';
        if (text.substr(pos + 1, 3) == "inh") {
            read_text += "inh";
            pos += 3;
            return Token::Sinh;
        }
        if (text.substr(pos + 1, 3) == "qrt") {
            read_text += "qrt";
            pos += 3;
            return Token::Sqrt;
        }
        if (text.substr(pos + 1, 2) == "gn") {
            read_text += "gn";
            pos += 2;
            return Token::Sgn;
        }
        if (text.substr(pos + 1, 2) == "in") {
            read_text += "in";
            pos += 2;
            return Token::Sin;
        }
        return Token::SymbolicConstant;
    }

    Token Scanner::try_read_tangent(std::string& read_text) {
        read_text += 't';
        if (text.substr(pos + 1, 2) == "gh") {
            read_text += "gh";
            pos += 2;
            return Token::Tanh;
        }
        if (text.substr(pos + 1, 1) == "g") {
            read_text += "g";
            ++pos;
            return Token::Tan;
        }
        if (text.substr(pos + 1, 3) == "anh") {
            read_text += "anh";
            pos += 3;
            return Token::Tanh;
        }
        if (text.substr(pos + 1, 2) == "an") {
            read_text += "an";
            pos += 2;
            return Token::Tan;
        }
        return Token::SymbolicConstant;
    }

    Token Scanner::try_read_integral(std::string& read_text) {
        read_text += 'i';
        if (text.substr(pos + 1, 8) == "ntegrate") {
            read_text += "ntegrate";
            pos += 8;
            return Token::Integral;
        }
        if (text.substr(pos + 1, 7) == "ntegral") {
            read_text += "ntegral";
            pos += 7;
            return Token::Integral;
        }
        if (text.substr(pos + 1, 2) == "nt") {
            read_text += "nt";
            pos += 2;
            return Token::Integral;
        }
        return Token::SymbolicConstant;
    }

    Token Scanner::read_after_start(Token& state, std::string& read_text) {
        if (pos == text.size() - 1) {
            read_text = "<end>";
            return Token::End;
        }
        switch (text[++pos]) {
        case '+':
            read_text += '+';
            return Token::Plus;
        case '-':
            read_text += '-';
            return Token::Minus;
        case '*':
            read_text += '*';
            return Token::Dot;
        case '/':
            read_text += '/';
            return Token::Dash;
        case '^':
            read_text += '^';
            return Token::Caret;
        case ' ':
            break;
        case '(':
            read_text += '(';
            return Token::OpenBrace;
        case ')':
            read_text += ')';
            return Token::CloseBrace;
        case '_':
            read_text += '_';
            return Token::Underscore;
        case '.':
            read_text += '.';
            state = Token::Double;
            break;
        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
            read_text += text[pos];
            state = Token::Integer;
            break;
        case 'x':
            read_text += 'x';
            state = Token::Variable;
            break;
        case 'e':
            read_text += 'e';
            state = Token::E;
            break;
        case 'p':
            read_text += 'p';
            if (text.substr(pos + 1, 1) == "i") {
                read_text += 'i';
                ++pos;
                state = Token::Pi;
            }
            else {
                state = Token::SymbolicConstant;
            }
            break;
        case 'a':
            state = try_read_abs_inverse_trig(read_text);
            break;
        case 'c':
            state = try_read_cosine_cotangent(read_text);
            break;
        case 'l':
            state = try_read_log(read_text);
            break;
        case 's':
            state = try_read_sign_sine_sqrt(read_text);
            break;
        case 't':
            state = try_read_tangent(read_text);
            break;
        case 'i':
            state = try_read_integral(read_text);
            break;
        case 'd':
            read_text += 'd';
            if (text.substr(pos + 1, 1) == "x") {
                read_text += 'x';
                ++pos;
                state = Token::Differential;
            }
            else {
                state = Token::SymbolicConstant;
            }
            break;
        default:
            read_text += text[pos];
            if (std::isalpha(text[pos]) != 0) {
                state = Token::SymbolicConstant;
                break;
            }
            read_letter_sequence(read_text);
            return Token::Error;
        }
        return Token::Start;
    }

    Token Scanner::check_if_no_letter_ahead(std::string& read_text, Token return_on_success) {
        if (pos == text.size() - 1 || std::isalpha(text[pos + 1]) == 0) {
            return return_on_success;
        }
        ++pos;
        read_letter_sequence(read_text);
        return Token::Error;
    }

    Token Scanner::scan(std::string& read_text) {
        Token state = Token::Start;
        Token returned = Token::Start;
        read_text = "";
        while (true) {
            switch (state) {
            case Token::Start:
                returned = read_after_start(state, read_text);
                if (returned == Token::Start) {
                    break;
                }
                return returned;
            case Token::Integer:
                if (pos == text.size() - 1) {
                    return Token::Integer;
                }
                switch (text[++pos]) {
                case '0':
                case '1':
                case '2':
                case '3':
                case '4':
                case '5':
                case '6':
                case '7':
                case '8':
                case '9':
                    read_text += text[pos];
                    state = Token::Integer;
                    break;
                case '.':
                    read_text += '.';
                    state = Token::Double;
                    break;
                default:
                    --pos;
                    return Token::Integer;
                }
                break;
            case Token::Double:
                if (pos == text.size() - 1) {
                    return Token::Double;
                }
                switch (text[++pos]) {
                case '0':
                case '1':
                case '2':
                case '3':
                case '4':
                case '5':
                case '6':
                case '7':
                case '8':
                case '9':
                    read_text += text[pos];
                    state = Token::Double;
                    break;
                case '.':
                    read_text += '.';
                    return Token::Error;
                default:
                    --pos;
                    return Token::Double;
                }
                break;
            case Token::Variable:
                return check_if_no_letter_ahead(read_text, Token::Variable);
            case Token::E:
                return check_if_no_letter_ahead(read_text, Token::E);
            case Token::Pi:
                return check_if_no_letter_ahead(read_text, Token::Pi);
            case Token::SymbolicConstant:
                return check_if_no_letter_ahead(read_text, Token::SymbolicConstant);
            case Token::Integral:
                return check_if_no_letter_ahead(read_text, Token::Integral);
            case Token::Differential:
                return check_if_no_letter_ahead(read_text, Token::Differential);
            default:
                if (isFunction(state)) {
                    return check_if_no_letter_ahead(read_text, state);
                }
                return Token::Error;
            }
        }
    }
}
