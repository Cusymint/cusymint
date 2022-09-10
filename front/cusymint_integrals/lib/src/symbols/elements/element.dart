import 'package:cusymint_integrals/src/symbols/symbol.dart';

abstract class Element extends Symbol {
  const Element({required this.toTexResult, required this.toUtfResult});

  final String toTexResult;
  final String toUtfResult;

  @override
  String toTex() {
    return toTexResult;
  }

  @override
  String toUtf() {
    return toUtfResult;
  }
}
