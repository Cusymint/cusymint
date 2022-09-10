import 'package:cusymint_integrals/src/symbols/elements/element.dart';

class EulerConstant extends Element {
  const EulerConstant() : super(toTexResult: 'e', toUtfResult: 'e');
}

class PiConstant extends Element {
  const PiConstant() : super(toTexResult: '\\pi', toUtfResult: 'π');
}
