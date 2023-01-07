#pragma once

#include <string>

class SolverProcessManager {
    private:
        std::string executable_name = "srvcusymint";
        std::string create_solve_command(const std::string& input) const;
        std::string create_solve_with_steps_command(const std::string& input) const;

    public:
        /*
         * @brief: Creates a new solver process and reads the solver output
         * in json format. When the solver process has an error, the errors
         * are set in the json output.
         * 
         * Note that we need to spawn a new process for each solver call
         * because any errors in CUDA runtime cannot be cleared otherwise.
         * Relevant SO question:
         * https://stackoverflow.com/questions/56329377/reset-cuda-context-after-exception
         * 
         * @param input: The input to the solver in raw utf-8 format
         * 
         * @return: The solver output in json format
         */
        std::string try_solve(const std::string& input) const;
        std::string try_solve_with_steps(const std::string& input) const;
};