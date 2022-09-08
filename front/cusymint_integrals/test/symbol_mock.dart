import 'package:cusymint_integrals/src/symbols/symbol.dart';

class SymbolMock extends Symbol {
  const SymbolMock(this.toTexResult, this.toUtfResult);

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

class LeftSymbolMock extends SymbolMock {
  const LeftSymbolMock() : super('left', 'left');
}

class RightSymbolMock extends SymbolMock {
  const RightSymbolMock() : super('right', 'right');
}
