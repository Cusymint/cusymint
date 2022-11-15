#include "JsonFormatter.cuh"
#include <fmt/core.h>

inline std::string escape_backslashes(const std::string& str) {
    std::string result;
    for (const auto& symbol : str) {
        if (symbol == '\\') {
            result += "\\\\";
        }
        else {
            result += symbol;
        }
    }
    return result;
}

inline std::string quote(const std::string& str) {
    auto escaped = escape_backslashes(str);
    return fmt::format("\"{}\"", escaped);
}

inline std::string generate_key_value(const std::string& key, const std::string& value) {
    return fmt::format("{}: {}", quote(key), quote(value));
}

std::string JsonFormatter::format(Expression* input, Expression* output) const {
    std::string json = "{";

    if (input != nullptr) {
        auto input_in_utf_pair = generate_key_value("inputInUtf", input->to_string());
        auto input_in_tex_pair = generate_key_value("inputInTex", input->to_tex());

        json += fmt::format("{}, {},", input_in_utf_pair, input_in_tex_pair);
    }

    if (output != nullptr) {
        auto output_in_utf_pair = generate_key_value("outputInUtf", output->to_string());
        auto output_in_tex_pair = generate_key_value("outputInTex", output->to_tex());

        json += fmt::format("{}, {},", output_in_utf_pair, output_in_tex_pair);
    }

    if (json.back() == ',') {
        json.pop_back();
    }

    json += "}";

    return json;
}
