#include "ResponseBuilder.cuh"

#include <fmt/core.h>
#include <fmt/format.h>

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

void ResponseBuilder::set_input(const Expression& expression) {
    auto input_in_utf_pair = generate_key_value("inputInUtf", expression.to_string());
    auto input_in_tex_pair = generate_key_value("inputInTex", expression.to_tex());
    input = fmt::format("{}, {},", input_in_utf_pair, input_in_tex_pair);
}

void ResponseBuilder::set_output(const Expression& expression) {
    auto output_in_utf_pair = generate_key_value("outputInUtf", expression.to_string());
    auto output_in_tex_pair = generate_key_value("outputInTex", expression.to_tex());
    output = fmt::format("{}, {},", output_in_utf_pair, output_in_tex_pair);
}

void ResponseBuilder::add_error(const std::string& error) {
    errors.push_back(quote(error));
}

std::string ResponseBuilder::get_errors_in_json() const {
    return fmt::format("errors: [{}]", fmt::join(errors, ", "));
}

std::string ResponseBuilder::get_json_response() const {
    std::string inside = fmt::format("{} {} {}", input, output, get_errors_in_json());

    if (inside.back() == ',') {
        inside.pop_back();
    }

    return fmt::format("{{{}}}", inside);
}
