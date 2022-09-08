import 'package:cusymint_integrals/src/symbols/operators/two_arguments_operator.dart';

class Division extends TwoArgumentsOperator {
  const Division(super.childLeft, super.childRight);

  @override
  String toTex() {
    return '\\frac{${childLeft.toTex()}}{${childRight.toTex()}}';
  }

  @override
  String toUtf() {
    return '${childLeft.toUtf()} ÷ ${childRight.toUtf()}';
  }
}
