#include "Parser.cuh"
#include <stdexcept>

bool isFunction(Token tok) { return tok >= Token::Asin; }

Parser::Parser(Scanner* scanner) : scanner(scanner) {}

void Parser::next_token() { tok = scanner->scan(read_text); }

void Parser::throw_error() { throw std::invalid_argument("Unexpected token: " + read_text); }

void Parser::match_and_get_next_token(Token token) {
    if (tok == token) {
        next_token();
    }
    else {
        throw_error();
    }
}

std::vector<Sym::Symbol> Parser::expr() {
    std::vector<Sym::Symbol> sum = term();
    while (tok == Token::Plus || tok == Token::Minus) {
        SymbolicOperator oper = tok == Token::Plus ? Sym::operator+ : static_cast<SymbolicOperator>(Sym::operator-);
        next_token();
        sum = oper(sum, term());
    }
    return sum;
}

std::vector<Sym::Symbol> Parser::term() {
    std::vector<Sym::Symbol> product = factor();
    while (tok == Token::Dot || tok == Token::Dash) {
        SymbolicOperator oper = tok == Token::Dot ? Sym::operator* : Sym::operator/;
        next_token();
        product = oper(product, factor());
    }
    return product;
}

std::vector<Sym::Symbol> Parser::factor() {
    std::vector<Sym::Symbol> fact = power_arg();
    if (tok == Token::Caret) {
        next_token();
        fact = fact ^ factor();
    }
    return fact;
}

std::vector<Sym::Symbol> Parser::power_arg() {
    std::vector<Sym::Symbol> internal_expression;
    std::string prev_text;
    switch (tok) {
    case Token::Integer:
    case Token::Double:
        prev_text = read_text;
        next_token();
        return Sym::num(std::stof(prev_text));
    case Token::SymbolicConstant:
        prev_text = read_text;
        next_token();
        return Sym::cnst(prev_text.c_str());
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
        internal_expression = expr();
        match_and_get_next_token(Token::CloseBrace); // )
        return internal_expression;
    case Token::Minus:
        next_token(); // -
        return -power_arg();
    default:
        if (isFunction(tok)) {
            SymbolicFunction func = function();
            match_and_get_next_token(Token::OpenBrace); // (
            internal_expression = expr();
            match_and_get_next_token(Token::CloseBrace); // )
            return func(internal_expression);
        }
        else {
            throw_error();
        }
        break;
    }
    return {};
}

SymbolicFunction Parser::function() {
    const SymbolicFunction empty = [](const std::vector<Sym::Symbol>& symbol) { return symbol; };
    const SymbolicFunction functions[] = {Sym::arcsin,   Sym::arccos, Sym::arctan, Sym::arccot,
                                          Sym::cos,      Sym::cosh,   Sym::coth,   /*ln*/ empty,
                                          /*log*/ empty, Sym::sin,    Sym::sinh,   /*sqrt*/ empty,
                                          Sym::tan,      Sym::tanh};
    Token prev = tok;
    next_token();
    return functions[static_cast<int>(prev) - static_cast<int>(Token::Asin)];
}

std::vector<Sym::Symbol> Parser::parse() {
    if (tok != Token::Start) {
        printf("ERROR: Parser has already processed a string.\n");
        return {};
    }
    next_token();
    try {
        std::vector<Sym::Symbol> e = expr();
        match_and_get_next_token(Token::End);
        return e;
    } catch (std::invalid_argument exc) {
        printf("ERROR: %s\n", exc.what());
        return {};
    }
}