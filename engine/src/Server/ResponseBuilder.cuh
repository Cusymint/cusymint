#pragma once

#include <string>
#include <vector>

#include "../Solver/Expression.cuh"

class ResponseBuilder {
    private:
        std::string input;
        std::string output;
        std::vector<std::string> errors;
        std::string history;
        bool has_input;
        bool has_output;
        std::string get_errors_in_json() const;

    public:
        void set_input(const Expression& expression);
        void set_output(const Expression& expression);
        void add_error(const std::string& error);
        void set_history(const std::vector<std::string>& history_entries);
        std::string get_json_response() const;
};