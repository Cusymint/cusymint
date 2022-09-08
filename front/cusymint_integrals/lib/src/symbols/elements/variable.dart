import 'package:cusymint_integrals/src/symbols/symbol.dart';

class Variable extends Symbol {
  const Variable();

  @override
  String toTex() {
    return 'x';
  }

  @override
  String toUtf() {
    return 'x';
  }
}
