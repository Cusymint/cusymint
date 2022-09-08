import 'package:cusymint_integrals/cusymint_integrals.dart';
import 'package:fluent_assertions/fluent_assertions.dart';
import 'package:test/test.dart';

import '../symbol_mock.dart';

void main() {
  test('toTex', () {
    final left = LeftSymbolMock();
    final right = RightSymbolMock();

    final division = Division(left, right);

    final result = division.toTex();

    result.shouldBeEqualTo('\\frac{left}{right}');
  });

  test('toString', () {
    final left = LeftSymbolMock();
    final right = RightSymbolMock();

    final division = Division(left, right);

    final result = division.toString();

    result.shouldBeEqualTo('left ÷ right');
  });

  test('toUtf', () {
    final left = LeftSymbolMock();
    final right = RightSymbolMock();

    final division = Division(left, right);

    final result = division.toUtf();

    result.shouldBeEqualTo('left ÷ right');
  });
}
