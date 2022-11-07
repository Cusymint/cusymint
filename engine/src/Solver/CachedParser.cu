#include "CachedParser.cuh"
#include "../Parser/Parser.cuh"
#include "../Parser/Scanner.cuh"

CachedParser::CachedParser() : cache() {}

std::optional<std::vector<Sym::Symbol>> CachedParser::get_from_cache(const std::string& key) {
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    return std::nullopt;
}

void CachedParser::add_to_cache(const std::string& key, const std::vector<Sym::Symbol>& value) {
    cache[key] = value;
}

std::vector<Sym::Symbol> CachedParser::get_parser_result(const std::string& input) {
    auto mutableInput = std::string(input);
    auto scanner = Scanner(mutableInput);
    auto parser = Parser(&scanner);
    auto result = parser.parse();
    return result;
}


std::vector<Sym::Symbol> CachedParser::parse(const std::string& input) {
    auto cached = get_from_cache(input);
    if (cached.has_value()) {
        return cached.value();
    }

    auto result = get_parser_result(input);
    add_to_cache(input, result);
    return result;
}
