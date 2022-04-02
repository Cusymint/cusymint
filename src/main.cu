#include <cstdlib>
#include <cstring>

#include <iostream>
#include <vector>

namespace Sym {
    enum class Type {
        // Basic types
        Unknown,
        Variable,
        NumericConstant, // rational constants with given numeric value, e.g. 5, 1.345, 12.44
        KnownConstant,   // well known real constants, e.g. pi, e (Euler's number)
        UnknownConstant, // constants marked with letters, e.g. a, phi, delta
        // Arithmetic
        Addition,
        Negative,
        Product,
        Reciprocal,
        Power,
        // Trigonometric functions
        Sine,
        Cosine,
        Tangent,
        Cotangent
    };

    enum class KnownConstantValue { Unknown, Pi, E };

    union Symbol;

#define DECLARE_SYMBOL(_name, _simple)    \
    struct _name {                        \
        Sym::Type type;                   \
        bool simplified;                  \
                                          \
        static _name create() {           \
            return {                      \
                .type = Sym::Type::_name, \
                .simplified = false,      \
            };                            \
        }

// A struct is POD iff it is standard-layout and trivial.
// standard-layout is required to guarantee that all symbolic types have the `type` member
// at offset 0 in the memory layout (necessary for later use in Symbol union).
// trivial is necessary for copying symbolic structs using simple memcpy calls
#define END_DECLARE_SYMBOL(_name) \
    }                             \
    ;                             \
    static_assert(std::is_pod<_name>::value, "Type '" #_name "' has to be POD, but is not!");

#define ONE_ARGUMENT_OP_SYMBOL size_t total_size;

#define TWO_ARGUMENT_OP_SYMBOL \
    size_t total_size;         \
    size_t second_arg_offset;

#define DEFINE_TO_STRING(_str) \
    std::string to_string() { return _str; }

    DECLARE_SYMBOL(Unknown, true)
    std::string to_string() { return "Unknown"; }
    END_DECLARE_SYMBOL(Unknown)

    DECLARE_SYMBOL(Variable, true)
    DEFINE_TO_STRING("x");
    END_DECLARE_SYMBOL(Variable)

    DECLARE_SYMBOL(NumericConstant, true)
    double value;

    DEFINE_TO_STRING(std::to_string(value));
    END_DECLARE_SYMBOL(NumericConstant)

    DECLARE_SYMBOL(KnownConstant, true)
    KnownConstantValue value;

    std::string to_string() {
        switch (value) {
        case KnownConstantValue::E:
            return "e";
        case KnownConstantValue::Pi:
            return "π";
        case KnownConstantValue::Unknown:
        default:
            return "<Undefined known constant>";
        }
    }
    END_DECLARE_SYMBOL(KnownConstant)

    DECLARE_SYMBOL(UnknownConstant, true)
    static constexpr size_t NAME_LEN = 8;
    char name[NAME_LEN];
    DEFINE_TO_STRING(name)
    END_DECLARE_SYMBOL(UnknownConstant)

    DECLARE_SYMBOL(Addition, false)
    TWO_ARGUMENT_OP_SYMBOL
    END_DECLARE_SYMBOL(Addition)

    DECLARE_SYMBOL(Negative, false)
    ONE_ARGUMENT_OP_SYMBOL
    END_DECLARE_SYMBOL(Negative)

    DECLARE_SYMBOL(Product, false)
    TWO_ARGUMENT_OP_SYMBOL
    END_DECLARE_SYMBOL(Product)

    DECLARE_SYMBOL(Reciprocal, false)
    ONE_ARGUMENT_OP_SYMBOL
    END_DECLARE_SYMBOL(Reciprocal)

    DECLARE_SYMBOL(Power, false)
    TWO_ARGUMENT_OP_SYMBOL
    END_DECLARE_SYMBOL(Power)

    DECLARE_SYMBOL(Sine, false)
    ONE_ARGUMENT_OP_SYMBOL
    END_DECLARE_SYMBOL(Sine)

    DECLARE_SYMBOL(Cosine, false)
    ONE_ARGUMENT_OP_SYMBOL
    END_DECLARE_SYMBOL(Cosine)

    DECLARE_SYMBOL(Tangent, false)
    ONE_ARGUMENT_OP_SYMBOL
    END_DECLARE_SYMBOL(Tangent)

    DECLARE_SYMBOL(Cotangent, false)
    ONE_ARGUMENT_OP_SYMBOL
    END_DECLARE_SYMBOL(Cotangent)

    union Symbol {
        Unknown unknown;
        Variable variable;
        NumericConstant numeric_constant;
        KnownConstant known_constant;
        UnknownConstant unknown_constant;
        Addition addition;
        Negative negative;
        Product product;
        Reciprocal reciprocal;
        Power power;
        Sine sine;
        Cosine cosine;
        Tangent tangent;
        Cotangent cotangent;

        __host__ __device__ inline Type type() const { return unknown.type; }
        __host__ __device__ inline bool is(Type type) const { return unknown.type == type; }
        template <class T> __host__ __device__ inline T& as() {
            return *reinterpret_cast<T*>(this);
        }

        __host__ __device__ bool is_simple_variable_polynomial() const {
            return this[0].is(Type::Power) && this[1].is(Type::Variable) &&
                   this[2].is(Type::NumericConstant) && this[2].numeric_constant.value != 0.0;
        }

        __host__ __device__ bool is_variable_reciprocal() const {
            return this[0].is(Type::Reciprocal) && this[1].is(Type::NumericConstant) &&
                   this[2].is(Type::Variable);
        }

        __host__ __device__ bool is_numeric_constant_addition() const {
            return this[0].is(Type::Addition) && this[1].is(Type::NumericConstant) &&
                   this[2].is(Type::NumericConstant);
        }

        __host__ __device__ bool is_numeric_constant_product() const {
            return this[0].is(Type::Product) && this[1].is(Type::NumericConstant) &&
                   this[2].is(Type::NumericConstant);
        }

        __host__ __device__ bool is_numeric_constant_negation() const {
            return this[0].is(Type::Negative) && this[1].is(Type::NumericConstant);
        }

        __host__ std::string to_string() {
            switch (unknown.type) {
            case Type::Unknown:
                return unknown.to_string();
            case Type::Variable:
                return variable.to_string();
            case Type::NumericConstant:
                return numeric_constant.to_string();
            case Type::KnownConstant:
                return known_constant.to_string();
            case Type::UnknownConstant:
                return unknown_constant.to_string();
            case Type::Addition:
                return "(" + this[1].to_string() + "+" +
                       this[addition.second_arg_offset].to_string() + ")";
            case Type::Negative:
                return "-(" + this[1].to_string() + ")";
            case Type::Product:
                return "(" + this[1].to_string() + "*" +
                       this[product.second_arg_offset].to_string() + ")";
            case Type::Reciprocal:
                return "(1/(" + this[1].to_string() + "))";
            case Type::Power:
                return "(" + this[1].to_string() + "^" + this[power.second_arg_offset].to_string() +
                       ")";
            case Type::Sine:
                return "sin(" + this[1].to_string() + ")";
            case Type::Cosine:
                return "cos(" + this[1].to_string() + ")";
            case Type::Tangent:
                return "tan(" + this[1].to_string() + ")";
            case Type::Cotangent:
                return "cot(" + this[1].to_string() + ")";
            }
        }
    };

    std::vector<Symbol> operator+(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs) {
        std::vector<Symbol> res(lhs.size() + rhs.size() + 1);
        res[0].addition = Addition::create();
        res[0].addition.total_size = res.size();
        res[0].addition.second_arg_offset = 1 + lhs.size();
        auto next = std::copy(lhs.begin(), lhs.end(), res.begin() + 1);
        std::copy(rhs.begin(), rhs.end(), next);

        return res;
    }

    std::vector<Symbol> operator-(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs) {
        std::vector<Symbol> res(lhs.size() + rhs.size() + 2);
        res[0].addition = Addition::create();
        res[0].addition.total_size = res.size();
        res[0].addition.second_arg_offset = 1 + lhs.size();
        auto next = std::copy(lhs.begin(), lhs.end(), res.begin() + 1);
        next->negative = Negative::create();
        next->negative.total_size = res.size();
        std::copy(rhs.begin(), rhs.end(), next + 1);

        return res;
    }

    std::vector<Symbol> operator-(const std::vector<Symbol>& arg) {
        std::vector<Symbol> res(arg.size() + 1);
        res[0].negative = Negative::create();
        res[0].negative.total_size = res.size();
        std::copy(arg.begin(), arg.end(), res.begin() + 1);

        return res;
    }

    std::vector<Symbol> operator*(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs) {
        std::vector<Symbol> res(lhs.size() + rhs.size() + 1);
        res[0].product = Product::create();
        res[0].product.total_size = res.size();
        res[0].product.second_arg_offset = 1 + lhs.size();
        auto next = std::copy(lhs.begin(), lhs.end(), res.begin() + 1);
        std::copy(rhs.begin(), rhs.end(), next);

        return res;
    }

    std::vector<Symbol> operator/(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs) {
        std::vector<Symbol> res(lhs.size() + rhs.size() + 2);
        res[0].product = Product::create();
        res[0].product.total_size = res.size();
        res[0].product.second_arg_offset = 1 + lhs.size();
        auto next = std::copy(lhs.begin(), lhs.end(), res.begin() + 1);
        next->reciprocal = Reciprocal::create();
        next->reciprocal.total_size = res.size();
        std::copy(rhs.begin(), rhs.end(), next + 1);

        return res;
    }

    std::vector<Symbol> operator^(const std::vector<Symbol>& lhs, const std::vector<Symbol>& rhs) {
        std::vector<Symbol> res(lhs.size() + rhs.size() + 1);
        res[0].power = Power::create();
        res[0].power.total_size = res.size();
        res[0].power.second_arg_offset = 1 + lhs.size();
        auto next = std::copy(lhs.begin(), lhs.end(), res.begin() + 1);
        std::copy(rhs.begin(), rhs.end(), next);

        return res;
    }

    std::vector<Symbol> sin(const std::vector<Symbol>& arg) {
        std::vector<Symbol> res(arg.size() + 1);
        res[0].sine = Sine::create();
        res[0].sine.total_size = res.size();
        std::copy(arg.begin(), arg.end(), res.begin() + 1);

        return res;
    }

    std::vector<Symbol> cos(const std::vector<Symbol>& arg) {
        std::vector<Symbol> res(arg.size() + 1);
        res[0].cosine = Cosine::create();
        res[0].cosine.total_size = res.size();
        std::copy(arg.begin(), arg.end(), res.begin() + 1);

        return res;
    }

    std::vector<Symbol> tan(const std::vector<Symbol>& arg) {
        std::vector<Symbol> res(arg.size() + 1);
        res[0].tangent = Tangent::create();
        res[0].tangent.total_size = res.size();
        std::copy(arg.begin(), arg.end(), res.begin() + 1);

        return res;
    }

    std::vector<Symbol> cot(const std::vector<Symbol>& arg) {
        std::vector<Symbol> res(arg.size() + 1);
        res[0].cotangent = Cotangent::create();
        res[0].cotangent.total_size = res.size();
        std::copy(arg.begin(), arg.end(), res.begin() + 1);

        return res;
    }

    std::vector<Symbol> known_constant(KnownConstantValue value) {
        std::vector<Symbol> v(1);
        v[0].known_constant = KnownConstant::create();
        v[0].known_constant.value = value;
        return v;
    }

    std::vector<Symbol> e() { return known_constant(KnownConstantValue::E); }

    std::vector<Symbol> pi() { return known_constant(KnownConstantValue::Pi); }

    std::vector<Symbol> var() {
        std::vector<Symbol> v(1);
        v[0].variable = Variable::create();
        return v;
    }

    std::vector<Symbol> cnst(const char name[UnknownConstant::NAME_LEN]) {
        std::vector<Symbol> v(1);
        v[0].unknown_constant = UnknownConstant::create();
        std::copy(name, name + UnknownConstant::NAME_LEN, v[0].unknown_constant.name);
        return v;
    }

    std::vector<Symbol> num(double value) {
        std::vector<Symbol> v(1);
        v[0].numeric_constant = NumericConstant::create();
        v[0].numeric_constant.value = value;
        return v;
    }

} // namespace Sym

int main() {
    std::cout << "Creating an expression" << std::endl;
    std::vector<Sym::Symbol> expression =
        Sym::num(10) + (Sym::e() ^ (-Sym::pi())) - Sym::cos(Sym::var()) / Sym::sin(Sym::cnst("a"));
    std::cout << "Expression created" << std::endl;

    std::cout << "Printing expression" << std::endl;
    std::cout << expression[0].to_string() << std::endl;
    std::cout << "Expression printed" << std::endl;
}
