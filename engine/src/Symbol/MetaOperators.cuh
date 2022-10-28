#ifndef META_OPERATORS_CUH
#define META_OPERATORS_CUH

#include "Symbol.cuh"

#include "Utils/Meta.cuh"

namespace Sym {
    struct Copy {
        using AdditionalArgs = cuda::std::tuple<cuda::std::reference_wrapper<const Symbol>>;
        __host__ __device__ static void init(Symbol& dst, const AdditionalArgs& args) {
            cuda::std::get<0>(args).get().copy_to(&dst);
        };
    };

    template <class Op, class Inner> struct OneArgOperator {
        using AdditionalArgs = typename Inner::AdditionalArgs;

        __host__ __device__ static void init(Symbol& dst,
                                             const AdditionalArgs& additional_args = {}) {
            Op* const operator_ = dst << Op::builder();
            Inner::init(operator_->arg(), additional_args);
            operator_->seal();
        };
    };

    template <class Op, class LInner, class RInner> struct TwoArgOperator {
        using LAdditionalArgs = typename LInner::AdditionalArgs;
        static constexpr size_t LAdditionalArgsSize = cuda::std::tuple_size_v<LAdditionalArgs>;

        using RAdditionalArgs = typename RInner::AdditionalArgs;
        static constexpr size_t RAdditionalArgsSize = cuda::std::tuple_size_v<RAdditionalArgs>;

        using AdditionalArgs = Util::TupleCat<LAdditionalArgs, RAdditionalArgs>;

        __host__ __device__ static void init(Symbol& dst, const AdditionalArgs& args = {}) {
            Op* const operator_ = dst << Op::builder();
            LInner::init(operator_->arg1(), Util::slice_tuple<0, LAdditionalArgsSize>(args));
            operator_->seal_arg1();
            RInner::init(operator_->arg2(),
                         Util::slice_tuple<LAdditionalArgsSize, RAdditionalArgsSize>(args));
            operator_->seal();
        };
    };

    struct Var {
        using AdditionalArgs = cuda::std::tuple<>;
        __host__ __device__ static void init(Symbol& dst, const AdditionalArgs& /*args*/) {
            dst.init_from(Variable::create());
        };
    };

    struct Num {
        using AdditionalArgs = cuda::std::tuple<double>;
        __host__ __device__ static void init(Symbol& dst, const AdditionalArgs& args) {
            dst.init_from(NumericConstant::with_value(cuda::std::get<0>(args)));
        };
    };

    template <KnownConstantValue V> struct KnownConstantOperator {
        using AdditionalArgs = cuda::std::tuple<>;
        __host__ __device__ static void init(Symbol& dst, const AdditionalArgs& /*args*/) {
            dst.init_from(KnownConstant::with_value(V));
        };
    };

    using Pi = KnownConstantOperator<KnownConstantValue::Pi>;
    using E = KnownConstantOperator<KnownConstantValue::E>;

    struct Const {
        using AdditionalArgs = cuda::std::tuple<const char[UnknownConstant::NAME_LEN]>;
        __host__ __device__ static void init(Symbol& dst, const AdditionalArgs& args) {
            dst.init_from(UnknownConstant::create());
            Util::copy_mem(dst.as<UnknownConstant>().name, cuda::std::get<0>(args),
                           UnknownConstant::NAME_LEN);
        };
    };

    template <class Inner> struct SolutionOfIntegral {
        using IAdditionalArgs = typename Inner::AdditionalArgs;
        static constexpr size_t IAdditionalArgsSize = cuda::std::tuple_size_v<IAdditionalArgs>;

        using SolutionArgs = cuda::std::tuple<cuda::std::reference_wrapper<const Integral>>;
        static constexpr size_t SolutionArgsSize = cuda::std::tuple_size_v<SolutionArgs>;

        using AdditionalArgs = Util::TupleCat<SolutionArgs, IAdditionalArgs>;

        __host__ __device__ static void init(Symbol& dst, const AdditionalArgs& args) {
            auto& integral = cuda::std::get<0>(args).get();
            auto* const solution = dst << Solution::builder();
            Symbol::copy_symbol_sequence(Symbol::from(solution->first_substitution()),
                                         Symbol::from(integral.first_substitution()),
                                         integral.substitutions_size());
            solution->seal_substitutions(integral.substitution_count,
                                         integral.substitutions_size());

            Inner::init(*solution->expression(),
                        Util::slice_tuple<SolutionArgsSize, IAdditionalArgsSize>(args));

            solution->seal();
        }
    };

    template <class Inner> struct Candidate {
        using IAdditionalArgs = typename Inner::AdditionalArgs;
        static constexpr size_t IAdditionalArgsSize = cuda::std::tuple_size_v<IAdditionalArgs>;

        using CandidateArgs = cuda::std::tuple<cuda::std::tuple<size_t, size_t, unsigned>>;
        static constexpr size_t CandidateArgsSize = cuda::std::tuple_size_v<CandidateArgs>;

        using AdditionalArgs = Util::TupleCat<CandidateArgs, IAdditionalArgs>;

        __host__ __device__ static void init(Symbol& dst, const AdditionalArgs& args) {
            auto* const candidate = dst << SubexpressionCandidate::builder();
            candidate->vacancy_expression_idx = cuda::std::get<0>(cuda::std::get<0>(args));
            candidate->vacancy_idx = cuda::std::get<0>(cuda::std::get<1>(args));
            candidate->subexpressions_left = cuda::std::get<0>(cuda::std::get<2>(args));

            Inner::init(candidate->arg(),
                        Util::slice_tuple<CandidateArgsSize, IAdditionalArgsSize>(args));

            candidate->seal();
        }
    };

    struct Vacancy {
        using AdditionalArgs = cuda::std::tuple<size_t, size_t, int>;

        __host__ __device__ static void init(Symbol& dst, const AdditionalArgs& args) {
            dst.init_from(SubexpressionVacancy::create());
            auto& vacancy = dst.as<SubexpressionVacancy>();
            vacancy.candidate_expression_count = cuda::std::get<0>(args);
            vacancy.candidate_integral_count = cuda::std::get<1>(args);
            vacancy.is_solved = cuda::std::get<2>(args);
        }
    };

    template <class L, class R> using Add = TwoArgOperator<Addition, L, R>;
    template <class I> using Neg = OneArgOperator<Negation, I>;

    template <class L, class R> using Mul = TwoArgOperator<Product, L, R>;
    template <class I> using Inv = OneArgOperator<Reciprocal, I>;

    template <class L, class R> using Pow = TwoArgOperator<Power, L, R>;

    template <class I> using Sin = OneArgOperator<Sine, I>;
    template <class I> using Cos = OneArgOperator<Cosine, I>;
    template <class I> using Tan = OneArgOperator<Tangent, I>;
    template <class I> using Cot = OneArgOperator<Cotangent, I>;
    template <class I> using Arcsin = OneArgOperator<Arcsine, I>;
    template <class I> using Arccos = OneArgOperator<Arccosine, I>;
    template <class I> using Arctan = OneArgOperator<Arctangent, I>;
    template <class I> using Arccot = OneArgOperator<Arccotangent, I>;

    template <class I> using Ln = OneArgOperator<Logarithm, I>;
}

#endif
