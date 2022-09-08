import 'package:cusymint_integrals/src/symbols/operators/two_arguments_operator.dart';

class Multiplication extends TwoArgumentsOperator {
  const Multiplication(super.childLeft, super.childRight);

  @override
  String toTex() {
    return '${childLeft.toTex()} \\cdot ${childRight.toTex()}';
  }

  @override
  String toUtf() {
    return '${childLeft.toUtf()} * ${childRight.toUtf()}';
  }
}
