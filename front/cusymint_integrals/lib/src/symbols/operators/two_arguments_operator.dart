import 'package:cusymint_integrals/src/symbols/symbol.dart';

abstract class TwoArgumentsOperator extends Symbol {
  const TwoArgumentsOperator(this.childLeft, this.childRight);

  final Symbol childLeft;
  final Symbol childRight;
}
