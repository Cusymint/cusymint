#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <optional>

#include "Expression.cuh"

#include "../Symbol/Symbol.cuh"

class CachedParser {
    private:
        std::unordered_map<std::string, Expression> _cache;
        std::optional<Expression> _get_from_cache(const std::string& key);
        void _add_to_cache(const std::string& key, const Expression& value);
        Expression _get_parser_result(const std::string& key);

    public:
        CachedParser();
        Expression parse(const std::string& input);        
};
