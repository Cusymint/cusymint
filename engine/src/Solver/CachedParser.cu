#include "CachedParser.cuh"
#include "../Parser/Parser.cuh"
#include "../Parser/Scanner.cuh"

CachedParser::CachedParser() : _cache() {}

std::optional<Expression> CachedParser::_get_from_cache(const std::string& key) {
    auto it = _cache.find(key);
    if (it != _cache.end()) {
        return it->second;
    }
    return std::nullopt;
}

void CachedParser::_add_to_cache(const std::string& key, const Expression& value) {
    _cache.insert({key, value});
}

Expression CachedParser::_get_parser_result(const std::string& input) {
    auto mutable_input = std::string(input);
    auto scanner = Scanner(mutable_input);
    auto parser = Parser(&scanner);
    auto result = parser.parse();
    return Expression(result);
}


Expression CachedParser::parse(const std::string& input) {
    auto cached = _get_from_cache(input);
    if (cached.has_value()) {
        return cached.value();
    }

    auto result = _get_parser_result(input);
    _add_to_cache(input, result);
    return result;
}
