#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <optional>

#include "../Symbol/Symbol.cuh"

class CachedParser {
    private:
        std::unordered_map<std::string, std::vector<Sym::Symbol>> cache;
        std::optional<std::vector<Sym::Symbol>> get_from_cache(const std::string& key);
        void add_to_cache(const std::string& key, const std::vector<Sym::Symbol>& value);
        std::vector<Sym::Symbol> get_parser_result(const std::string& key);

    public:
        CachedParser();
        std::vector<Sym::Symbol> parse(const std::string& input);        
};
