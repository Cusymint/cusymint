#include "Scanner.hpp"

bool isLetter(char c)
{
	return c >= 'a' && c <= 'z';
}

Scanner::Scanner(std::string text)
{
	this->text = text;
}

Token Scanner::scan(std::string &read_text)
{
	Token state = Token::Start;
	read_text = "";
	while (true)
	{
		switch (state)
		{
		case Token::Start:
			if (pos == text.size() - 1)
			{
				read_text = "<end>";
				return Token::End;
			}
			switch (text[++pos])
			{
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
                if (text.substr(pos+1,1)=="i")
                {
                    read_text += 'i';
                    ++pos;
                    state = Token::Pi;
                }
                else
                {
                    state = Token::SymbolicConstant;
                }
                break;
			case 'a':
				read_text += 'a';
				if (text.substr(pos + 1, 5) == "rcsin")
				{
					read_text += "rcsin";
					pos += 5;
					return Token::Asin;
				}
				if (text.substr(pos + 1, 5) == "rccos")
				{
					read_text += "rccos";
					pos += 5;
					return Token::Acos;
				}
				if (text.substr(pos + 1, 4) == "rctg")
				{
					read_text += "rctg";
					pos += 4;
					return Token::Atan;
				}
				if (text.substr(pos + 1, 5) == "rcctg")
				{
					read_text += "rcctg";
					pos += 5;
					return Token::Acot;
				}
				state = Token::SymbolicConstant;
				break;
			case 'c':
				read_text += 'c';
				if (text.substr(pos + 1, 3) == "osh")
				{
					read_text += "osh";
					pos += 3;
					return Token::Cosh;
				}
				if (text.substr(pos + 1, 3) == "tgh")
				{
					read_text += "tgh";
					pos += 3;
					return Token::Coth;
				}
				if (text.substr(pos + 1, 3) == "os")
				{
					read_text += "os";
					pos += 2;
					return Token::Cos;
				}
				if (text.substr(pos + 1, 3) == "tg")
				{
					read_text += "tg";
					pos += 2;
					return Token::Cot;
				}
				state = Token::SymbolicConstant;
				break;
			case 'l':
				read_text += 'l';
				if (text.substr(pos + 1, 2) == "og")
				{
					read_text += "og";
					pos += 2;
					return Token::Log;
				}
				if (text.substr(pos + 1, 1) == "n")
				{
					read_text += "n";
					++pos;
					return Token::Ln;
				}
				state = Token::SymbolicConstant;
				break;
			case 's':
				read_text += 's';
				if (text.substr(pos + 1, 3) == "inh")
				{
					read_text += "inh";
					pos += 3;
					return Token::Sinh;
				}
				if (text.substr(pos + 1, 3) == "qrt")
				{
					read_text += "qrt";
					pos += 3;
					return Token::Sqrt;
				}
				if (text.substr(pos + 1, 2) == "in")
				{
					read_text += "in";
					pos += 2;
					return Token::Sin;
				}
				state = Token::SymbolicConstant;
				break;
			case 't':
				read_text += 't';
				if (text.substr(pos + 1, 2) == "gh")
				{
					read_text += "gh";
					pos += 2;
					return Token::Tanh;
				}
				if (text.substr(pos + 1, 1) == "g")
				{
					read_text += "g";
					++pos;
					return Token::Tan;
				}
				state = Token::SymbolicConstant;
				break;
			default:
				read_text += text[pos]; 
				if (isLetter(text[pos]))
				{
					state = Token::SymbolicConstant;
					break;
				}
				return Token::Error;
			}
			break;
		case Token::Integer:
			if (pos == text.size() - 1) return Token::Integer;
			switch (text[++pos])
			{
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
			if (pos == text.size() - 1) return Token::Double;
			switch (text[++pos])
			{
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
			if (pos == text.size() - 1) return Token::Variable;
			if (isLetter(text[++pos]))
			{
				read_text += text[pos];
				return Token::Error;
			}
			--pos;
			return Token::Variable;
        case Token::E:
            if (pos == text.size() - 1 || !isLetter(text[pos+1])) return Token::E;
            else
            {
                read_text += text[++pos];
                return Token::Error;
            }
        case Token::Pi:
            if (pos == text.size() - 1 || !isLetter(text[pos+1])) return Token::Pi;
            else
            {
                read_text += text[++pos];
                return Token::Error;
            }
		case Token::SymbolicConstant:
			if (pos == text.size() - 1) return Token::SymbolicConstant;
			if (isLetter(text[++pos]))
			{
				read_text += text[pos];
				return Token::Error;
			}
			--pos;
			return Token::SymbolicConstant;
		default:
			return Token::Error;
		}
		
	}
}