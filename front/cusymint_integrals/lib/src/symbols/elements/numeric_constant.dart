import 'package:cusymint_integrals/src/symbols/symbol.dart';

class NumericConstant extends Symbol {
  const NumericConstant(this.value);

  final double value;

  @override
  String toTex() {
    if (value.floor() == value) {
      return value.toInt().toString();
    } else {
      return value.toString();
    }
  }

  @override
  String toUtf() {
    return toTex();
  }
}
