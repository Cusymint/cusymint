import 'package:cusymint_integrals/src/symbols/operators/two_arguments_operator.dart';

class Substraction extends TwoArgumentsOperator {
  const Substraction(super.childLeft, super.childRight);

  @override
  String toTex() {
    return '${childLeft.toTex()} - ${childRight.toTex()}';
  }

  @override
  String toUtf() {
    return '${childLeft.toUtf()} - ${childRight.toUtf()}';
  }
}
