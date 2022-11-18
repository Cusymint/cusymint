#ifndef META_OPERATORS_CUH
#define META_OPERATORS_CUH

#include "Symbol.cuh"

#include <type_traits>

#include "Symbol/TreeIterator.cuh"
#include "Utils/Meta.cuh"

#define DEFINE_GET_SAME                                                    \
    template <typename U = void, std::enable_if_t<HAS_SAME, U>* = nullptr> \
    __host__ __device__ static const Symbol& get_same(const Symbol& dst)

namespace Sym {
    using Unsized = Util::MetaEmpty<size_t>;
    template <size_t S> using Size = Util::MetaOptional<S>;
    using SingletonSize = Size<1>;
    template <size_t S, class... Others>
    using SizeFrom = Util::MetaOptionalsSum<Size<S>, typename Others::Size...>;

    struct Copy {
        using AdditionalArgs = cuda::std::tuple<cuda::std::reference_wrapper<const Symbol>>;
        using Size = Unsized;

        static constexpr bool HAS_SAME = false;

        __host__ __device__ static void init(Symbol& dst, const AdditionalArgs& args) {
            cuda::std::get<0>(args).get().copy_to(&dst);
        };
    };

    template <class T, class U> struct PatternPair {
        using Size = Unsized;

        __host__ __device__ static bool match_pair(const Symbol& expr1, const Symbol& expr2) {
            if constexpr (T::HAS_SAME) {
                return T::match(expr1) && U::match(expr2, T::get_same(expr1));
            }
            else {
                return T::match(expr1) && U::match(expr2);
            }
        }
    };

    struct Same {
        using AdditionalArgs = cuda::std::tuple<>;
        using Size = Unsized;

        static constexpr bool HAS_SAME = true;

        __host__ __device__ static const Symbol& get_same(const Symbol& dst) { return dst; }

        __host__ __device__ static bool match(const Symbol&) { return true; }

        __host__ __device__ static bool match(const Symbol& dst, const Symbol& other_same) {
            return Symbol::compare_trees(dst, other_same);
        }
    };

    template <class Head, class... Tail>
    struct FirstHavingSame : std::conditional_t<Head::HAS_SAME, Head, FirstHavingSame<Tail...>> {};
    template <class T> struct FirstHavingSame<T> : T {};

    template <class... Matchers> struct AnyOf {
        using AdditionalArgs = cuda::std::tuple<>;
        using Size = Unsized;

        static constexpr bool HAS_SAME = (Matchers::HAS_SAME || ...);

        DEFINE_GET_SAME { return FirstHavingSame<Matchers...>::get_same(dst); }

        __host__ __device__ static bool match(const Symbol& dst, const Symbol& other_same) {
            return (Matchers::match(dst, other_same) || ...);
        }

        __host__ __device__ static bool match(const Symbol& dst) {
            return (Matchers::match(dst) || ...);
        };
    };

    template <class... Matchers> struct AllOf {
        using AdditionalArgs = cuda::std::tuple<>;
        using Size = Unsized;

        static constexpr bool HAS_SAME = (Matchers::HAS_SAME || ...);

        DEFINE_GET_SAME { return FirstHavingSame<Matchers...>::get_same(dst); }

        __host__ __device__ static bool match(const Symbol& dst, const Symbol& other_same) {
            return (Matchers::match(dst, other_same) && ...);
        }

        __host__ __device__ static bool match(const Symbol& dst) {
            return (Matchers::match(dst) && ...);
        };
    };

    template <class Inner> struct Not {
        using AdditionalArgs = typename Inner::AdditionalArgs;
        using Size = Unsized;

        static constexpr bool HAS_SAME = Inner::HAS_SAME;

        DEFINE_GET_SAME { return Inner::get_same(dst); }

        __host__ __device__ static bool match(const Symbol& dst, const Symbol& other_same) {
            return !Inner::match(dst, other_same);
        };
        __host__ __device__ static bool match(const Symbol& dst) { return !Inner::match(dst); };
    };

    struct Any {
        using AdditionalArgs = cuda::std::tuple<>;
        using Size = Unsized;

        static constexpr bool HAS_SAME = false;

        __host__ __device__ static bool match(const Symbol& /*dst*/) { return true; };
        __host__ __device__ static bool match(const Symbol& /*dst*/, const Symbol&) {
            return true;
        };
    };

    template <class Op, class Inner> struct OneArgOperator {
        using AdditionalArgs = typename Inner::AdditionalArgs;
        using Size = SizeFrom<1, Inner>;

        static constexpr bool HAS_SAME = Inner::HAS_SAME;

        DEFINE_GET_SAME { return Inner::get_same(dst.as<Op>().arg()); }

        __host__ __device__ static void init(Symbol& dst, const AdditionalArgs& additional_args) {
            Op* const operator_ = dst << Op::builder();
            Inner::init(operator_->arg(), additional_args);
            operator_->seal();
        };

        template <typename U = void,
                  std::enable_if_t<Inner::Size::HAS_VALUE && Util::is_empty_tuple<AdditionalArgs>,
                                   U>* = nullptr>
        __host__ __device__ static void init(Symbol& dst) {
            init(dst, {});
        };

        __host__ __device__ static bool match(const Symbol& dst, const Symbol& other_same) {
            return dst.is(Op::TYPE) && Inner::match(dst.as<Op>().arg(), other_same);
        }

        __host__ __device__ static bool match(const Symbol& dst) {
            return dst.is(Op::TYPE) && Inner::match(dst.as<Op>().arg());
        }
    };

    template <class Op, class LInner, class RInner> struct TwoArgOperator {
        using LAdditionalArgs = typename LInner::AdditionalArgs;
        static constexpr size_t L_ADDITIONAL_ARGS_SIZE = cuda::std::tuple_size_v<LAdditionalArgs>;

        using RAdditionalArgs = typename RInner::AdditionalArgs;
        static constexpr size_t R_ADDITIONAL_ARGS_SIZE = cuda::std::tuple_size_v<RAdditionalArgs>;

        using AdditionalArgs = Util::TupleCat<LAdditionalArgs, RAdditionalArgs>;
        static constexpr bool HAS_SAME = LInner::HAS_SAME || RInner::HAS_SAME;

        using Size = SizeFrom<1, LInner, RInner>;

        DEFINE_GET_SAME {
            if constexpr (LInner::HAS_SAME) {
                return LInner::get_same(dst.as<Op>().arg1());
            }
            else {
                return RInner::get_same(dst.as<Op>().arg2());
            }
        }

        __host__ __device__ static void init(Symbol& dst, const AdditionalArgs& args) {
            Op* const operator_ = dst << Op::builder();
            LInner::init(operator_->arg1(), Util::slice_tuple<0, L_ADDITIONAL_ARGS_SIZE>(args));
            operator_->seal_arg1();
            RInner::init(operator_->arg2(),
                         Util::slice_tuple<L_ADDITIONAL_ARGS_SIZE, R_ADDITIONAL_ARGS_SIZE>(args));
            operator_->seal();
        };

        template <typename U = void,
                  std::enable_if_t<LInner::Size::HAS_VALUE && RInner::Size::HAS_VALUE &&
                                       Util::is_empty_tuple<AdditionalArgs>,
                                   U>* = nullptr>
        __host__ __device__ static void init(Symbol& dst) {
            init(dst, {});
        };

        __host__ __device__ static bool match(const Symbol& dst, const Symbol& other_same) {
            return dst.is(Op::TYPE) && LInner::match(dst.as<Op>().arg1(), other_same) &&
                   RInner::match(dst.as<Op>().arg2(), other_same);
        }

        template <typename U = void,
                  std::enable_if_t<LInner::HAS_SAME && RInner::HAS_SAME, U>* = nullptr>
        __host__ __device__ static bool match(const Symbol& dst) {
            return dst.is(Op::TYPE) && LInner::match(dst.as<Op>().arg1()) &&
                   RInner::match(dst.as<Op>().arg2(), LInner::get_same(dst.as<Op>().arg1()));
        }

        template <typename U = void,
                  std::enable_if_t<!(LInner::HAS_SAME && RInner::HAS_SAME), U>* = nullptr>
        __host__ __device__ static bool match(const Symbol& dst) {
            return dst.is(Op::TYPE) && LInner::match(dst.as<Op>().arg1()) &&
                   RInner::match(dst.as<Op>().arg2());
        }
    };

    struct Var {
        using AdditionalArgs = cuda::std::tuple<>;
        using Size = SingletonSize;

        static constexpr bool HAS_SAME = false;

        __host__ __device__ static void init(Symbol& dst, const AdditionalArgs& /*args*/) {
            init(dst);
        };

        __host__ __device__ static void init(Symbol& dst) { dst.init_from(Variable::create()); };

        __host__ __device__ static bool match(const Symbol& dst) { return dst.is(Type::Variable); }

        __host__ __device__ static bool match(const Symbol& dst, const Symbol&) {
            return match(dst);
        }
    };

    struct Num {
        using AdditionalArgs = cuda::std::tuple<double>;
        using Size = SingletonSize;

        static constexpr bool HAS_SAME = false;

        __host__ __device__ static void init(Symbol& dst, const AdditionalArgs& args) {
            dst.init_from(NumericConstant::with_value(cuda::std::get<0>(args)));
        };

        __host__ __device__ static bool match(const Symbol& dst) {
            return dst.is(Type::NumericConstant);
        }
        __host__ __device__ static bool match(const Symbol& dst, const Symbol&) {
            return match(dst);
        }
    };

    struct Const {
        using AdditionalArgs = cuda::std::tuple<>;
        using Size = Unsized;

        static constexpr bool HAS_SAME = false;

        __host__ __device__ static bool match(const Symbol& dst) { return dst.is_constant(); }
        __host__ __device__ static bool match(const Symbol& dst, const Symbol&) {
            return match(dst);
        }
    };

    // In C++17, doubles can't be template parameters.
    template <int V> struct Integer {
        using AdditionalArgs = cuda::std::tuple<>;
        using Size = SingletonSize;

        static constexpr bool HAS_SAME = false;

        __host__ __device__ static void init(Symbol& dst, const AdditionalArgs& /*args*/) {
            init(dst);
        };

        __host__ __device__ static void init(Symbol& dst) {
            dst.init_from(NumericConstant::with_value(V));
        };

        __host__ __device__ static bool match(const Symbol& dst) {
            return dst.is(Type::NumericConstant) && dst.as<NumericConstant>().value == V;
        }

        __host__ __device__ static bool match(const Symbol& dst, const Symbol&) {
            return match(dst);
        }
    };

    template <KnownConstantValue V> struct KnownConstantOperator {
        using AdditionalArgs = cuda::std::tuple<>;
        using Size = SingletonSize;

        static constexpr bool HAS_SAME = false;

        __host__ __device__ static void init(Symbol& dst, const AdditionalArgs& /*args*/) {
            init(dst);
        };

        __host__ __device__ static void init(Symbol& dst) {
            dst.init_from(KnownConstant::with_value(V));
        };

        __host__ __device__ static bool match(const Symbol& dst) {
            return dst.is(Type::KnownConstant) && dst.as<KnownConstant>().value == V;
        }

        __host__ __device__ static bool match(const Symbol& dst, const Symbol&) {
            return match(dst);
        }
    };

    using Pi = KnownConstantOperator<KnownConstantValue::Pi>;
    using E = KnownConstantOperator<KnownConstantValue::E>;

    template <class Inner> struct SolutionOfIntegral {
        using IAdditionalArgs = typename Inner::AdditionalArgs;
        static constexpr size_t I_ADDITIONAL_ARGS_SIZE = cuda::std::tuple_size_v<IAdditionalArgs>;

        using SolutionArgs = cuda::std::tuple<cuda::std::reference_wrapper<const Integral>>;
        static constexpr size_t SOLUTION_ARGS_SIZE = cuda::std::tuple_size_v<SolutionArgs>;

        using AdditionalArgs = Util::TupleCat<SolutionArgs, IAdditionalArgs>;
        static constexpr bool HAS_SAME = Inner::HAS_SAME;

        using Size = Unsized;

        DEFINE_GET_SAME { return Inner::get_same(*dst.as<Solution>().expression()); }

        __host__ __device__ static void init(Symbol& dst, const AdditionalArgs& args) {
            auto& integral = cuda::std::get<0>(args).get();
            auto* const solution = dst << Solution::builder();
            Symbol::copy_symbol_sequence(Symbol::from(solution->first_substitution()),
                                         Symbol::from(integral.first_substitution()),
                                         integral.substitutions_size());
            solution->seal_substitutions(integral.substitution_count,
                                         integral.substitutions_size());

            Inner::init(*solution->expression(),
                        Util::slice_tuple<SOLUTION_ARGS_SIZE, I_ADDITIONAL_ARGS_SIZE>(args));

            solution->seal();
        }

        __host__ __device__ static bool match(const Symbol& dst) {
            return dst.is(Type::Solution) && Inner::match(dst.as<Solution>().expression());
        }

        __host__ __device__ static bool match(const Symbol& dst, const Symbol& other_same) {
            return dst.is(Type::Solution) &&
                   Inner::match(dst.as<Solution>().expression(), other_same);
        }
    };

    template <class Inner> struct Candidate {
        using IAdditionalArgs = typename Inner::AdditionalArgs;
        static constexpr size_t I_ADDITIONAL_ARGS_SIZE = cuda::std::tuple_size_v<IAdditionalArgs>;

        using CandidateArgs = cuda::std::tuple<cuda::std::tuple<size_t, size_t, unsigned>>;
        static constexpr size_t CANDIDATE_ARGS_SIZE = cuda::std::tuple_size_v<CandidateArgs>;

        using AdditionalArgs = Util::TupleCat<CandidateArgs, IAdditionalArgs>;
        static constexpr bool HAS_SAME = Inner::HAS_SAME;

        using Size = SizeFrom<1, Inner>;

        DEFINE_GET_SAME { return Inner::get_same(dst.as<SubexpressionCandidate>().arg()); }

        __host__ __device__ static void init(Symbol& dst, const AdditionalArgs& args) {
            auto* const candidate = dst << SubexpressionCandidate::builder();
            candidate->vacancy_expression_idx = cuda::std::get<0>(cuda::std::get<0>(args));
            candidate->vacancy_idx = cuda::std::get<1>(cuda::std::get<0>(args));
            candidate->subexpressions_left = cuda::std::get<2>(cuda::std::get<0>(args));

            Inner::init(candidate->arg(),
                        Util::slice_tuple<CANDIDATE_ARGS_SIZE, I_ADDITIONAL_ARGS_SIZE>(args));

            candidate->seal();
        }

        __host__ __device__ static bool match(const Symbol& dst, const Symbol& other_same) {
            return dst.is(Type::SubexpressionCandidate) &&
                   Inner::match(dst.as<SubexpressionCandidate>().arg(), other_same);
        }

        __host__ __device__ static bool match(const Symbol& dst) {
            return dst.is(Type::SubexpressionCandidate) &&
                   Inner::match(dst.as<SubexpressionCandidate>().arg());
        }
    };

    template <class Inner> struct Int {
        using IAdditionalArgs = typename Inner::AdditionalArgs;
        static constexpr size_t I_ADDITIONAL_ARGS_SIZE = cuda::std::tuple_size_v<IAdditionalArgs>;

        using IntegralArgs = cuda::std::tuple<cuda::std::reference_wrapper<const Integral>>;
        static constexpr size_t INTEGRAL_ARGS_SIZE = cuda::std::tuple_size_v<IntegralArgs>;

        using AdditionalArgs = Util::TupleCat<IntegralArgs, IAdditionalArgs>;
        static constexpr bool HAS_SAME = Inner::HAS_SAME;

        using Size = Unsized;

        DEFINE_GET_SAME { return Inner::get_same(*dst.as<Integral>().integrand()); }

        __host__ __device__ static void init(Symbol& dst, const AdditionalArgs& args) {
            cuda::std::get<0>(args).get().copy_without_integrand_to(&dst);
            auto& dst_integral = dst.as<Integral>();

            Inner::init(*dst_integral.integrand(),
                        Util::slice_tuple<INTEGRAL_ARGS_SIZE, I_ADDITIONAL_ARGS_SIZE>(args));
            dst_integral.seal();
        }

        __host__ __device__ static bool match(const Symbol& dst) {
            return dst.is(Type::Integral) && Inner::match(*dst.as<Integral>().integrand());
        }

        __host__ __device__ static bool match(const Symbol& dst, const Symbol& other_same) {
            return dst.is(Type::Integral) &&
                   Inner::match(*dst.as<Integral>().integrand(), other_same);
        }
    };

    struct Vacancy {
        using AdditionalArgs = cuda::std::tuple<size_t, size_t, int>;
        using Size = SingletonSize;

        static constexpr bool HAS_SAME = false;

        __host__ __device__ static void init(Symbol& dst, const AdditionalArgs& args) {
            dst.init_from(SubexpressionVacancy::create());
            auto& vacancy = dst.as<SubexpressionVacancy>();
            vacancy.candidate_expression_count = cuda::std::get<0>(args);
            vacancy.candidate_integral_count = cuda::std::get<1>(args);
            vacancy.is_solved = cuda::std::get<2>(args);
        }
    };

    struct SingleIntegralVacancy {
        using AdditionalArgs = cuda::std::tuple<>;
        using Size = SingletonSize;

        static constexpr bool HAS_SAME = false;

        __host__ __device__ static void init(Symbol& dst, const AdditionalArgs& /*args*/) {
            dst.init_from(SubexpressionVacancy::for_single_integral());
        }

        __host__ __device__ static bool match(const Symbol& dst) {
            return dst.is(Type::SubexpressionVacancy);
        }

        __host__ __device__ static bool match(const Symbol& dst, const Symbol&) {
            return match(dst);
        }
    };

    template <class L, class R> using Add = TwoArgOperator<Addition, L, R>;
    template <class I> using Neg = OneArgOperator<Negation, I>;
    template <class L, class R> using Sub = Add<L, Neg<R>>;

    template <class Head, class... Tail> struct Sum : Add<Head, Sum<Tail...>> {};
    template <class T> struct Sum<T> : T {};

    template <class L, class R> using Mul = TwoArgOperator<Product, L, R>;
    template <class I> using Inv = OneArgOperator<Reciprocal, I>;
    template <class L, class R> using Frac = Mul<L, Inv<R>>;

    template <class Head, class... Tail> struct Prod : Mul<Head, Prod<Tail...>> {};
    template <class T> struct Prod<T> : T {};

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

    /*
     * @brief Encapsulates procedure of creating `TwoArgOp` symbol tree from existing `SymbolTree`,
     * where every leaf of a tree (term of sum/factor of a product) is mapped by function of type
     * `OneArgOp`. Note that `TwoArgOp` and `SymbolTree` may be different types.
     */
    template <class SymbolTree> struct From {
        template <class TwoArgOp> struct Create {
            template <class OneArgOp> struct WithMap {
                using AdditionalArgs = cuda::std::tuple<
                    cuda::std::tuple<cuda::std::reference_wrapper<SymbolTree>, size_t>>;
                static constexpr bool HAS_SAME = false;

                __host__ __device__ static void init(Symbol& dst, const AdditionalArgs& args = {}) {
                    SymbolTree& tree = cuda::std::get<0>(cuda::std::get<0>(args));
                    size_t count = cuda::std::get<1>(cuda::std::get<0>(args));
                    Symbol* terms = &dst + count - 1;
                    TreeIterator<SymbolTree> iterator(&tree);
                    while (iterator.is_valid()) {
                        OneArgOp* operator_ = terms << OneArgOp::builder();
                        iterator.current()->copy_to(&operator_->arg());
                        operator_->seal();
                        terms += terms->size();
                        iterator.advance();
                    }
                    for (ssize_t i = static_cast<ssize_t>(count) - 2; i >= 0; --i) {
                        TwoArgOp* const operator_ = &dst + i << TwoArgOp::builder();
                        operator_->seal_arg1();
                        operator_->seal();
                    }
                }
            };
        };
    };
}

#endif
