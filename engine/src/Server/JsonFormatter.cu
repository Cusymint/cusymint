#include "JsonFormatter.cuh"

std::string JsonFormatter::format(Expression* input, Expression* output) const {
    std::string json = "{";

    if(input != nullptr) {
        json += "\"inputInUtf\":";
        json += input->to_string();
        json += ",";

        json += "\"inputInTex\":";
        json += input->to_tex();
        json += ",";
    }

    if(output != nullptr) {
        json += "\"outputInUtf\":";
        json += output->to_string();
        json += ",";

        json += "\"outputInTex\":";
        json += output->to_tex();
        json += ",";
    }

    if(json.back() == ',') {
        json.pop_back();
    }

    json += "}";

    return json;
}
