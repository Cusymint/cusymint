#include "Integrate.cuh"

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "Heuristic/Heuristic.cuh"
#include "StaticFunctions.cuh"

#include "Utils/Cuda.cuh"
#include "Utils/Meta.cuh"

namespace Sym {
    namespace {
        constexpr size_t TRANSFORM_GROUP_SIZE = 32;
        constexpr size_t MAX_EXPRESSION_COUNT = 256;
        constexpr size_t EXPRESSION_MAX_SYMBOL_COUNT = 256;

        /*
         * @brief Podejmuje próbę ustawienia `expressions[potential_solver_idx]` (będącego
         * SubexpressionCandidate) jako rozwiązania wskazywanego przez siebie SubexpressionVacancy
         *
         * @param expressions Talbica wyrażeń z kandydatem do rozwiązania i brakującym podwyrażeniem
         * @param potential_solver_idx Indeks kandydata do rozwiązania
         *
         * @return `false` jeśli nie udało się ustawić wybranego kandydata jako rozwiązanie
         * podwyrażenia, lub udało się, ale w nadwyrażeniu są jeszcze inne nierozwiązane
         * podwyrażenia. `true` jeśli się udało i było to ostatnie nierozwiązane podwyrażenie w
         * nadwyrażeniu.
         */
        __device__ bool try_set_solver_idx(Sym::ExpressionArray<>& expressions,
                                           const size_t potential_solver_idx) {
            const size_t& vacancy_expr_idx =
                expressions[potential_solver_idx].subexpression_candidate.vacancy_expression_idx;

            const size_t& vacancy_idx =
                expressions[potential_solver_idx].subexpression_candidate.vacancy_idx;

            Sym::SubexpressionVacancy& subexpr_vacancy =
                expressions[vacancy_expr_idx][vacancy_idx].subexpression_vacancy;

            const bool solver_lock_acquired = atomicCAS(&subexpr_vacancy.is_solved, 0, 1) == 0;

            if (!solver_lock_acquired) {
                return false;
            }

            subexpr_vacancy.solver_idx = potential_solver_idx;

            if (!expressions[vacancy_expr_idx].is(Sym::Type::SubexpressionCandidate)) {
                return true;
            }

            unsigned int subexpressions_left = atomicSub(
                &expressions[vacancy_expr_idx].subexpression_candidate.subexpressions_left, 1);

            return subexpressions_left == 0;
        }

        /*
         * @brief Sets `var` to `val` atomically
         *
         * @brief var Variable to set
         * @brief val Value assigned to `var`
         *
         * @return `false` if `var` was already equal to `val`, `true` otherwise
         */
        template <class T> __device__ bool try_set(T& var, const T& val) {
            const unsigned int previous_val = atomicExch(&var, val);
            return previous_val != val;
        }

        /*
         * @brief Gets target index from `scan` inclusive scan array at `index` index
         */
        __device__ uint32_t index_from_scan(const Util::DeviceArray<uint32_t>& scan,
                                            const size_t index) {
            if (index == 0) {
                return 0;
            }

            return scan[index - 1];
        }

        using KnownIntegralCheck = size_t (*)(const Integral* const integral);

        using KnownIntegralTransform = void (*)(const Integral* const integral,
                                                Symbol* const destination,
                                                Symbol* const help_space);

        __device__ size_t is_single_variable(const Integral* const integral) {
            return integral->integrand()->is(Type::Variable) ? 1 : 0;
        }

        __device__ size_t is_simple_variable_power(const Integral* const integral) {
            const Symbol* const integrand = integral->integrand();
            if (!integrand[0].is(Type::Power) || !integrand[1].is(Type::Variable)) {
                return 0;
            }

            if (integrand[2].is(Type::NumericConstant) &&
                integrand[2].as<NumericConstant>().value == -1.0) {
                return 0;
            }

            return integrand[2].is_constant() ? 1 : 0;
        }
        __device__ size_t is_variable_exponent(const Integral* const integral) {
            const Symbol* const integrand = integral->integrand();
            return integrand[0].is(Type::Power) && integrand[1].is(Type::KnownConstant) &&
                           integrand[1].as<KnownConstant>().value == KnownConstantValue::E &&
                           integrand[2].is(Type::Variable)
                       ? 1
                       : 0;
        }
        __device__ size_t is_simple_sine(const Integral* const integral) {
            const Symbol* const integrand = integral->integrand();
            return integrand[0].is(Type::Sine) && integrand[1].is(Type::Variable) ? 1 : 0;
        }

        __device__ size_t is_simple_cosine(const Integral* const integral) {
            const Symbol* const integrand = integral->integrand();
            return integrand[0].is(Type::Cosine) && integrand[1].is(Type::Variable) ? 1 : 0;
        }

        __device__ size_t is_constant(const Integral* const integral) {
            const Symbol* const integrand = integral->integrand();
            return integrand->is_constant() ? 1 : 0;
        }

        __device__ size_t is_known_arctan(const Integral* const integral) {
            const Symbol* const integrand = integral->integrand();
            // 1/(x^2+1) or 1/(1+x^2)
            return integrand[0].is(Type::Product) && integrand[1].is(Type::NumericConstant) &&
                           integrand[1].numeric_constant.value == 1.0 &&
                           integrand[2].is(Type::Reciprocal) && integrand[3].is(Type::Addition) &&
                           ((integrand[4].is(Type::Power) && integrand[5].is(Type::Variable) &&
                             integrand[6].is(Type::NumericConstant) &&
                             integrand[6].numeric_constant.value == 2.0 &&
                             integrand[7].is(Type::NumericConstant) &&
                             integrand[7].numeric_constant.value == 1.0) ||
                            (integrand[4].is(Type::NumericConstant) &&
                             integrand[4].numeric_constant.value == 1.0 &&
                             integrand[5].is(Type::Power) && integrand[6].is(Type::Variable) &&
                             integrand[7].is(Type::NumericConstant) &&
                             integrand[7].numeric_constant.value == 2.0))
                       ? 1
                       : 0;
        }

        /*
         * @brief Creates `Solution` and writes it to `destination` together with substitutions from
         * `integral`
         *
         * @param integral Integral from which substitutions are to be copied
         * @param destination Result destination
         *
         * @return Pointer to the symbol behind the last substitution in the result
         */
        __device__ Symbol* prepare_solution(const Integral* const integral,
                                            Symbol* const destination) {
            Solution* const solution = destination << Solution::builder();
            Symbol::copy_symbol_sequence(Symbol::from(solution->first_substitution()),
                                         Symbol::from(integral->first_substitution()),
                                         integral->substitutions_size());
            solution->seal_substitutions(integral->substitution_count,
                                         integral->substitutions_size());

            return solution->expression();
        }

        __device__ void integrate_single_variable(const Integral* const integral,
                                                  Symbol* const destination,
                                                  Symbol* const /*help_space*/) {
            Symbol* const solution_expr = prepare_solution(integral, destination);

            Product* const product = solution_expr << Product::builder();
            product->arg1().numeric_constant = NumericConstant::with_value(0.5);
            product->seal_arg1();

            Power* const power = &product->arg2() << Power::builder();
            power->arg1().variable = Variable::create();
            power->seal_arg1();
            power->arg2().numeric_constant = NumericConstant::with_value(2.0);
            power->seal();
            product->seal();

            destination->solution.seal();
        }

        __device__ void integrate_simple_variable_power(const Integral* const integral,
                                                        Symbol* const destination,
                                                        Symbol* const /*help_space*/) {
            const Symbol* const integrand = integral->integrand();

            Symbol* const solution_expr = prepare_solution(integral, destination);
            const Symbol* const exponent = &integral->integrand()->power.arg2();

            // 1/(c+1) * x^(c+1), c może być całym drzewem
            Product* const product = solution_expr << Product::builder();

            Reciprocal* const reciprocal = &product->arg1() << Reciprocal::builder();
            Addition* const multiplier_addition = &reciprocal->arg() << Addition::builder();
            exponent->copy_to(&multiplier_addition->arg1());
            multiplier_addition->seal_arg1();
            multiplier_addition->arg2().numeric_constant = NumericConstant::with_value(1.0);
            multiplier_addition->seal();
            reciprocal->seal();
            product->seal_arg1();

            Power* const power = &product->arg2() << Power::builder();
            power->arg1().variable = Variable::create();
            power->seal_arg1();
            Addition* const exponent_addition = &power->arg2() << Addition::builder();
            exponent->copy_to(&exponent_addition->arg1());
            exponent_addition->seal_arg1();
            exponent_addition->arg2().numeric_constant = NumericConstant::with_value(1.0);
            exponent_addition->seal();
            power->seal();
            product->seal();

            destination->solution.seal();
        }

        __device__ void integrate_variable_exponent(const Integral* const integral,
                                                    Symbol* const destination,
                                                    Symbol* const /*help_space*/) {
            Symbol* const solution_expr = prepare_solution(integral, destination);
            const Symbol* const integrand = integral->integrand();

            Power* const power = solution_expr << Power::builder();
            power->arg1().known_constant = KnownConstant::with_value(KnownConstantValue::E);
            power->seal_arg1();
            power->arg2().variable = Variable::create();
            power->seal();

            destination->solution.seal();
        }

        __device__ void integrate_simple_sine(const Integral* const integral,
                                              Symbol* const destination,
                                              Symbol* const /*help_space*/) {
            Symbol* const solution_expr = prepare_solution(integral, destination);
            const Symbol* const integrand = integral->integrand();

            Negation* const minus = solution_expr << Negation::builder();
            Cosine* const cos = &minus->arg() << Cosine::builder();
            cos->arg().variable = Variable::create();
            cos->seal();
            minus->seal();

            destination->solution.seal();
        }

        __device__ void integrate_simple_cosine(const Integral* const integral,
                                                Symbol* const destination,
                                                Symbol* const /*help_space*/) {
            Symbol* const solution_expr = prepare_solution(integral, destination);
            const Symbol* const integrand = integral->integrand();

            Sine* const sine = solution_expr << Sine::builder();
            sine->arg().variable = Variable::create();
            sine->seal();

            destination->solution.seal();
        }

        __device__ void integrate_constant(const Integral* const integral,
                                           Symbol* const destination,
                                           Symbol* const /*help_space*/) {
            const Symbol* const integrand = integral->integrand();
            Symbol* const solution_expr = prepare_solution(integral, destination);

            Product* const product = solution_expr << Product::builder();
            product->arg1().variable = Variable::create();
            product->seal_arg1();
            integrand->copy_to(&product->arg2());
            product->seal();

            destination->solution.seal();
        }

        __device__ void integrate_arctan(const Integral* const integral, Symbol* const destination,
                                         Symbol* const /*help_space*/) {
            const Symbol* const integrand = integral->integrand();
            Symbol* const solution_expr = prepare_solution(integral, destination);

            Arctangent* const arctangent = solution_expr << Arctangent::builder();
            arctangent->arg().variable = Variable::create();
            arctangent->seal();

            destination->solution.seal();
        }

        __device__ const KnownIntegralCheck KNOWN_INTEGRAL_CHECKS[] = {
            is_single_variable, is_simple_variable_power, is_variable_exponent,
            is_simple_sine,     is_simple_cosine,         is_constant,
            is_known_arctan,
        };

        __device__ const KnownIntegralTransform KNOWN_INTEGRAL_APPLICATIONS[] = {
            integrate_single_variable, integrate_simple_variable_power, integrate_variable_exponent,
            integrate_simple_sine,     integrate_simple_cosine,         integrate_constant,
            integrate_arctan,
        };

        constexpr size_t KNOWN_INTEGRAL_COUNT =
            Util::ensure_same_v<Util::array_len(KNOWN_INTEGRAL_CHECKS),
                                Util::array_len(KNOWN_INTEGRAL_APPLICATIONS)>;

    }

    __device__ bool is_nonzero(const size_t index,
                               const Util::DeviceArray<uint32_t>& inclusive_scan) {
        return index == 0 && inclusive_scan[index] != 0 ||
               index != 0 && inclusive_scan[index - 1] != inclusive_scan[index];
    }

    __global__ void simplify(ExpressionArray<> expressions, ExpressionArray<> help_spaces) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        for (size_t expr_idx = thread_idx; expr_idx < expressions.size();
             expr_idx += thread_count) {
            expressions[expr_idx].simplify(help_spaces.at(expr_idx));
        }
    }

    __global__ void
    check_for_known_integrals(const ExpressionArray<SubexpressionCandidate> integrals,
                              Util::DeviceArray<uint32_t> applicability) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        const size_t check_step = thread_count / TRANSFORM_GROUP_SIZE;

        for (size_t check_idx = thread_idx / TRANSFORM_GROUP_SIZE; check_idx < KNOWN_INTEGRAL_COUNT;
             check_idx += check_step) {
            for (size_t int_idx = thread_idx % TRANSFORM_GROUP_SIZE; int_idx < integrals.size();
                 int_idx += TRANSFORM_GROUP_SIZE) {
                size_t appl_idx = MAX_EXPRESSION_COUNT * check_idx + int_idx;
                applicability[appl_idx] =
                    KNOWN_INTEGRAL_CHECKS[check_idx](&integrals[int_idx].arg().as<Integral>());
            }
        }
    }

    __global__ void apply_known_integrals(const ExpressionArray<SubexpressionCandidate> integrals,
                                          ExpressionArray<> expressions,
                                          ExpressionArray<> help_spaces,
                                          const Util::DeviceArray<uint32_t> applicability) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        const size_t trans_step = thread_count / TRANSFORM_GROUP_SIZE;

        for (size_t trans_idx = thread_idx / TRANSFORM_GROUP_SIZE; trans_idx < KNOWN_INTEGRAL_COUNT;
             trans_idx += trans_step) {
            for (size_t int_idx = thread_idx % TRANSFORM_GROUP_SIZE; int_idx < integrals.size();
                 int_idx += TRANSFORM_GROUP_SIZE) {
                const size_t appl_idx = MAX_EXPRESSION_COUNT * trans_idx + int_idx;

                if (!is_nonzero(appl_idx, applicability)) {
                    continue;
                }

                const size_t dest_idx =
                    expressions.size() + index_from_scan(applicability, appl_idx);

                auto* const subexpr_candidate = expressions.at(dest_idx)
                                                << SubexpressionCandidate::builder();
                subexpr_candidate->copy_metadata_from(integrals[int_idx]);
                KNOWN_INTEGRAL_APPLICATIONS[trans_idx](&integrals[int_idx].arg().as<Integral>(),
                                                       &subexpr_candidate->arg(),
                                                       help_spaces.at(dest_idx));
                subexpr_candidate->seal();

                try_set_solver_idx(expressions, dest_idx);
            }
        }
    }

    __global__ void propagate_solved_subexpressions(ExpressionArray<> expressions) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        // W każdym węźle drzewa zależności wyrażeń zaczyna jeden wątek. Jeśli jego węzeł jest
        // rozwiązany, to próbuje się ustawić jako rozwiązanie swojego podwyrażenia w rodzicu. Jeśli
        // mu się to uda i nie pozostaną w rodzicu inne nierozwiązane podwyrażenia, to przechodzi do
        // niego i powtaża wszystko. W skrócie następuje propagacja informacji o rozwiązaniu z dołu
        // drzewa na samą górę.

        // Na expr_idx = 0 jest tylko SubexpressionVacancy oryginalnej całki, więc pomijamy
        for (size_t expr_idx = thread_idx + 1; expr_idx < expressions.size();
             expr_idx += thread_count) {
            size_t current_expr_idx = expr_idx;
            while (current_expr_idx != 0) {
                if (expressions[current_expr_idx].subexpression_candidate.subexpressions_left !=
                    0) {
                    break;
                }

                if (!try_set_solver_idx(expressions, current_expr_idx)) {
                    break;
                }

                // Przechodzimy w drzewie zależności do rodzica. Być może będziemy tam razem z
                // wątkiem, który tam zaczął pętlę. `try_set_solver_idx` jest jednak atomowe, więc
                // tylko jednemu z wątków uda się ustawić `solver_idx` na kolejnym rodzicu, więc
                // tylko jeden wątek tam przetrwa.
                current_expr_idx =
                    expressions[current_expr_idx].subexpression_candidate.vacancy_expression_idx;
            }
        }
    }

    __global__ void find_redundand_expressions(const ExpressionArray<> expressions,
                                               Util::DeviceArray<uint32_t> removability) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        // Look further and further in the dependency tree and check whether we are not trying to
        // solve something that has been solved already
        for (size_t expr_idx = thread_idx; expr_idx < expressions.size();
             expr_idx += thread_count) {
            removability[expr_idx] = 1;
            size_t current_expr_idx = expr_idx;

            while (current_expr_idx != 0) {
                const size_t& parent_idx =
                    expressions[current_expr_idx].subexpression_candidate.vacancy_expression_idx;
                const size_t& parent_vacancy_idx =
                    expressions[current_expr_idx].subexpression_candidate.vacancy_idx;
                const SubexpressionVacancy& parent_vacancy =
                    expressions[parent_idx][parent_vacancy_idx].subexpression_vacancy;

                if (parent_vacancy.is_solved == 1 &&
                    parent_vacancy.solver_idx != current_expr_idx) {
                    removability[expr_idx] = 0;
                    break;
                }

                current_expr_idx = parent_idx;
            }
        }
    }

    __global__ void
    find_redundand_integrals(const ExpressionArray<> integrals, const ExpressionArray<> expressions,
                             const Util::DeviceArray<uint32_t> expressions_removability,
                             Util::DeviceArray<uint32_t> integrals_removability) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        for (size_t int_idx = thread_idx; int_idx < integrals.size(); int_idx += thread_count) {
            const size_t& vacancy_expr_idx =
                integrals[int_idx].subexpression_candidate.vacancy_expression_idx;
            const size_t& vacancy_idx = integrals[int_idx].subexpression_candidate.vacancy_idx;

            const bool parent_expr_failed = expressions_removability[vacancy_expr_idx] == 0;
            const bool parent_vacancy_solved =
                expressions[vacancy_expr_idx][vacancy_idx].subexpression_vacancy.is_solved == 1;

            integrals_removability[int_idx] = parent_expr_failed || parent_vacancy_solved ? 0 : 1;
        }
    }

    __global__ void remove_expressions(const ExpressionArray<> expressions,
                                       const Util::DeviceArray<uint32_t> removability,
                                       ExpressionArray<> destinations) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        for (size_t expr_idx = thread_idx; expr_idx < expressions.size();
             expr_idx += thread_count) {
            if (!is_nonzero(expr_idx, removability)) {
                continue;
            }

            Symbol& destination = destinations[removability[expr_idx] - 1];
            expressions[expr_idx].copy_to(&destination);

            destination.if_is_do<SubexpressionCandidate>([&removability](auto& dst) {
                dst.vacancy_expression_idx = removability[dst.vacancy_expression_idx] - 1;
            });

            for (size_t symbol_idx = 0; symbol_idx < destination.size(); ++symbol_idx) {
                destination[symbol_idx].if_is_do<SubexpressionVacancy>([&removability](auto& vac) {
                    if (vac.is_solved != 1) {
                        return;
                    }

                    vac.solver_idx = removability[vac.solver_idx] - 1;
                    vac.candidate_integral_count = 0;
                });
            }
        }
    }

    __global__ void remove_integrals(const ExpressionArray<SubexpressionCandidate> integrals,
                                     const Util::DeviceArray<uint32_t> integrals_removability,
                                     const Util::DeviceArray<uint32_t> expressions_removability,
                                     ExpressionArray<> destinations) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        for (size_t int_idx = thread_idx; int_idx < integrals.size(); int_idx += thread_count) {
            if (!is_nonzero(int_idx, integrals_removability)) {
                continue;
            }

            Symbol& destination = destinations[integrals_removability[int_idx] - 1];
            integrals[int_idx].symbol()->copy_to(&destination);

            size_t& vacancy_expr_idx =
                destination.as<SubexpressionCandidate>().vacancy_expression_idx;
            vacancy_expr_idx = expressions_removability[vacancy_expr_idx] - 1;
        }
    }

    __global__ void are_integrals_failed(const Util::DeviceArray<uint32_t> expressions_removability,
                                         const ExpressionArray<SubexpressionCandidate> integrals,
                                         Util::DeviceArray<uint32_t> integrals_removability) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        for (size_t int_idx = thread_idx; int_idx < integrals.size(); int_idx += thread_count) {
            const size_t expr_idx = integrals[int_idx].vacancy_expression_idx;
            integrals_removability[int_idx] = expressions_removability[expr_idx];
        }
    }

    __global__ void
    check_heuristics_applicability(const ExpressionArray<SubexpressionCandidate> integrals,
                                   ExpressionArray<> expressions,
                                   Util::DeviceArray<uint32_t> new_integrals_flags,
                                   Util::DeviceArray<uint32_t> new_expressions_flags) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        const size_t check_step = thread_count / TRANSFORM_GROUP_SIZE;

        for (size_t check_idx = thread_idx / TRANSFORM_GROUP_SIZE; check_idx < Heuristic::COUNT;
             check_idx += check_step) {
            for (size_t int_idx = thread_idx % TRANSFORM_GROUP_SIZE; int_idx < integrals.size();
                 int_idx += TRANSFORM_GROUP_SIZE) {
                size_t appl_idx = MAX_EXPRESSION_COUNT * check_idx + int_idx;
                Heuristic::CheckResult result =
                    Heuristic::CHECKS[check_idx](&integrals[int_idx].arg().as<Integral>());
                new_integrals_flags[appl_idx] = result.new_integrals;
                new_expressions_flags[appl_idx] = result.new_expressions;

                const size_t& vacancy_expr_idx = integrals[int_idx].vacancy_expression_idx;
                const size_t& vacancy_idx = integrals[int_idx].vacancy_idx;
                SubexpressionVacancy& parent_vacancy =
                    expressions[vacancy_expr_idx][vacancy_idx].subexpression_vacancy;

                if (result.new_expressions == 0) {
                    // Assume new integrals are direct children of the vacancy
                    atomicAdd(&parent_vacancy.candidate_integral_count, result.new_integrals);
                }
                else {
                    // Assume new integrals are going to be children of new expressions, which are
                    // going to be children of the vacancy
                    atomicAdd(&parent_vacancy.candidate_expression_count, result.new_expressions);
                }
            }
        }
    }

    __global__ void apply_heuristics(const ExpressionArray<SubexpressionCandidate> integrals,
                                     ExpressionArray<> integrals_destinations,
                                     ExpressionArray<> expressions_destinations,
                                     ExpressionArray<> help_spaces,
                                     const Util::DeviceArray<uint32_t> new_integrals_indices,
                                     const Util::DeviceArray<uint32_t> new_expressions_indices) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        const size_t trans_step = thread_count / TRANSFORM_GROUP_SIZE;

        for (size_t trans_idx = thread_idx / TRANSFORM_GROUP_SIZE; trans_idx < Heuristic::COUNT;
             trans_idx += trans_step) {
            for (size_t int_idx = thread_idx % TRANSFORM_GROUP_SIZE; int_idx < integrals.size();
                 int_idx += TRANSFORM_GROUP_SIZE) {
                const size_t appl_idx = MAX_EXPRESSION_COUNT * trans_idx + int_idx;
                if (!is_nonzero(appl_idx, new_integrals_indices)) {
                    continue;
                }

                const size_t int_dst_idx = index_from_scan(new_integrals_indices, appl_idx);

                if (new_expressions_indices[appl_idx] != 0) {
                    const size_t expr_dst_idx = expressions_destinations.size() +
                                                index_from_scan(new_expressions_indices, appl_idx);
                    Heuristic::APPLICATIONS[trans_idx](
                        integrals[int_idx], integrals_destinations.iterator(int_dst_idx),
                        expressions_destinations.iterator(expr_dst_idx), help_spaces[int_dst_idx]);
                }
                else {
                    Heuristic::APPLICATIONS[trans_idx](
                        integrals[int_idx], integrals_destinations.iterator(int_dst_idx),
                        ExpressionArray<>::Iterator::Null(), help_spaces[int_dst_idx]);
                }
            }
        }
    }

    __global__ void propagate_failures_upwards(ExpressionArray<> expressions,
                                               Util::DeviceArray<uint32_t> failures) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        // expressions[0] has no parents, so nothing to update there
        for (size_t expr_idx = thread_idx + 1; expr_idx < expressions.size();
             expr_idx += thread_count) {
            SubexpressionCandidate& self_candidate = expressions[expr_idx].subexpression_candidate;

            // Some other thread was here already
            if (failures[expr_idx] == 0) {
                continue;
            }

            bool is_failed = false;

            // expressions[current_expr_idx][0] is subexpression_candidate, so it can be skipped
            for (size_t sym_idx = 1; sym_idx < expressions[expr_idx].size(); ++sym_idx) {
                if (!expressions[expr_idx][sym_idx].is(Type::SubexpressionVacancy)) {
                    continue;
                }

                SubexpressionVacancy& vacancy =
                    expressions[expr_idx][sym_idx].subexpression_vacancy;

                if (vacancy.candidate_integral_count == 0 &&
                    vacancy.candidate_expression_count == 0 && vacancy.is_solved == 0) {
                    is_failed = true;
                    break;
                }
            }

            if (!is_failed || !try_set(failures[expr_idx], 0U)) {
                continue;
            }

            size_t current_expr_idx = expr_idx;

            while (current_expr_idx != 0) {
                const size_t& parent_idx =
                    expressions[current_expr_idx].subexpression_candidate.vacancy_expression_idx;
                const size_t& vacancy_idx =
                    expressions[current_expr_idx].subexpression_candidate.vacancy_idx;
                SubexpressionVacancy& parent_vacancy =
                    expressions[parent_idx][vacancy_idx].subexpression_vacancy;

                if (parent_vacancy.candidate_integral_count != 0 || parent_vacancy.is_solved == 1) {
                    break;
                }

                const size_t parent_vacancy_candidates_left =
                    atomicSub(&parent_vacancy.candidate_expression_count, 1) - 1;

                // Go upwards if parent is failed
                if (parent_vacancy_candidates_left != 0 || !try_set(failures[parent_idx], 0U)) {
                    break;
                }

                current_expr_idx = parent_idx;
            }
        }
    }

    __global__ void propagate_failures_downwards(ExpressionArray<> expressions,
                                                 Util::DeviceArray<uint32_t> failures) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        // Top expression has no parents, so we skip it
        for (size_t expr_idx = thread_idx + 1; expr_idx < expressions.size();
             expr_idx += thread_count) {
            size_t current_expr_idx = expr_idx;

            while (current_expr_idx != 0) {
                const size_t& parent_idx =
                    expressions[current_expr_idx].subexpression_candidate.vacancy_expression_idx;

                if (failures[parent_idx] == 0) {
                    failures[expr_idx] = 0;
                    break;
                }

                current_expr_idx = parent_idx;
            }
        }
    }

    __global__ void
    find_redundand_integrals(const ExpressionArray<> integrals,
                             const Util::DeviceArray<uint32_t> expressions_removability,
                             Util::DeviceArray<uint32_t> integrals_removability) {
        const size_t thread_count = Util::thread_count();
        const size_t thread_idx = Util::thread_idx();

        for (size_t int_idx = thread_idx; int_idx < integrals.size(); int_idx += thread_count) {
            const size_t& parent_idx =
                integrals[int_idx].subexpression_candidate.vacancy_expression_idx;

            integrals_removability[int_idx] = expressions_removability[parent_idx];
        }
    }

    std::optional<std::vector<std::vector<Symbol>>>
    solve_integral(const std::vector<Symbol>& integral) {
        static constexpr size_t BLOCK_SIZE = 512;
        static constexpr size_t BLOCK_COUNT = 32;
        const size_t MAX_CHECK_COUNT =
            KNOWN_INTEGRAL_COUNT > Heuristic::COUNT ? KNOWN_INTEGRAL_COUNT : Heuristic::COUNT;
        const size_t SCAN_ARRAY_SIZE = MAX_CHECK_COUNT * MAX_EXPRESSION_COUNT;

        ExpressionArray<> expressions({single_integral_vacancy()}, EXPRESSION_MAX_SYMBOL_COUNT,
                                      MAX_EXPRESSION_COUNT);
        ExpressionArray<> expressions_swap(MAX_EXPRESSION_COUNT, EXPRESSION_MAX_SYMBOL_COUNT,
                                           expressions.size());

        ExpressionArray<SubexpressionCandidate> integrals({first_expression_candidate(integral)},
                                                          MAX_EXPRESSION_COUNT,
                                                          EXPRESSION_MAX_SYMBOL_COUNT);
        ExpressionArray<SubexpressionCandidate> integrals_swap(MAX_EXPRESSION_COUNT,
                                                               EXPRESSION_MAX_SYMBOL_COUNT);
        ExpressionArray<> help_spaces(MAX_EXPRESSION_COUNT, EXPRESSION_MAX_SYMBOL_COUNT,
                                      integrals.size());
        Util::DeviceArray<uint32_t> scan_array_1(SCAN_ARRAY_SIZE, true);
        Util::DeviceArray<uint32_t> scan_array_2(SCAN_ARRAY_SIZE, true);

        for (size_t i = 0;; ++i) {
            simplify<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, help_spaces);
            cudaDeviceSynchronize();

            check_for_known_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, scan_array_1);
            cudaDeviceSynchronize();

            thrust::inclusive_scan(thrust::device, scan_array_1.begin(), scan_array_1.end(),
                                   scan_array_1.data());
            cudaDeviceSynchronize();

            apply_known_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, expressions, help_spaces,
                                                               scan_array_1);
            cudaDeviceSynchronize();
            expressions.increment_size_from_device(scan_array_1.last());

            propagate_solved_subexpressions<<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions);
            cudaDeviceSynchronize();

            std::vector<Symbol> first_expression = expressions.to_vector(0);
            if (first_expression.data()->as<SubexpressionVacancy>().is_solved == 1) {
                // TODO: Collapse the tree instead of returning it
                simplify<<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions, help_spaces);
                return expressions.to_vector();
            }

            scan_array_1.zero_mem();
            find_redundand_expressions<<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions, scan_array_1);
            cudaDeviceSynchronize();

            scan_array_2.zero_mem(); // TODO: Not necessary?
            find_redundand_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, expressions,
                                                                  scan_array_1, scan_array_2);
            cudaDeviceSynchronize();

            thrust::inclusive_scan(thrust::device, scan_array_1.begin(), scan_array_1.end(),
                                   scan_array_1.data());
            thrust::inclusive_scan(thrust::device, scan_array_2.begin(), scan_array_2.end(),
                                   scan_array_2.data());
            cudaDeviceSynchronize();

            remove_expressions<true>
                <<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions, scan_array_1, expressions_swap);
            remove_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, scan_array_2, scan_array_1,
                                                          integrals_swap);
            cudaDeviceSynchronize();

            std::swap(expressions, expressions_swap);
            std::swap(integrals, integrals_swap);
            expressions.resize_from_device(scan_array_1.last());
            integrals.resize_from_device(scan_array_2.last());

            scan_array_1.zero_mem();
            scan_array_2.zero_mem();
            check_heuristics_applicability<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, expressions,
                                                                        scan_array_1, scan_array_2);
            cudaDeviceSynchronize();

            thrust::inclusive_scan(thrust::device, scan_array_1.begin(), scan_array_1.end(),
                                   scan_array_1.data());
            thrust::inclusive_scan(thrust::device, scan_array_2.begin(), scan_array_2.end(),
                                   scan_array_2.data());
            cudaDeviceSynchronize();

            apply_heuristics<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, integrals_swap, expressions,
                                                          help_spaces, scan_array_1, scan_array_2);
            cudaDeviceSynchronize();

            std::swap(integrals, integrals_swap);
            integrals.resize_from_device(scan_array_1.last());
            expressions.increment_size_from_device(scan_array_2.last());

            scan_array_1.set_mem(1);
            cudaDeviceSynchronize();

            propagate_failures_upwards<<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions, scan_array_1);
            cudaDeviceSynchronize();

            // First expression in the array has failed, all is lost
            if (scan_array_1.to_cpu(0) == 0) {
                return std::nullopt;
            }

            propagate_failures_downwards<<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions, scan_array_1);
            cudaDeviceSynchronize();

            scan_array_2.zero_mem();
            find_redundand_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, scan_array_1,
                                                                  scan_array_2);
            cudaDeviceSynchronize();

            thrust::inclusive_scan(thrust::device, scan_array_1.begin(), scan_array_1.end(),
                                   scan_array_1.data());
            thrust::inclusive_scan(thrust::device, scan_array_2.begin(), scan_array_2.end(),
                                   scan_array_2.data());
            cudaDeviceSynchronize();

            remove_expressions<false>
                <<<BLOCK_COUNT, BLOCK_SIZE>>>(expressions, scan_array_1, expressions_swap);
            remove_integrals<<<BLOCK_COUNT, BLOCK_SIZE>>>(integrals, scan_array_2, scan_array_1,
                                                          integrals_swap);
            cudaDeviceSynchronize();

            std::swap(expressions, expressions_swap);
            std::swap(integrals, integrals_swap);

            // How many expressions are left, cannot take scan_array_1.last() because space
            // after last place in scan_array_1 that corresponds to an expression is occupied
            // by ones
            expressions.resize(scan_array_1.to_cpu(expressions_swap.size() - 1));
            integrals.resize_from_device(scan_array_2.last());
            cudaDeviceSynchronize();

            scan_array_1.zero_mem();
            scan_array_2.zero_mem();
        }

        return std::nullopt;
    }

}
