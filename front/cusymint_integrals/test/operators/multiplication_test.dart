import 'package:cusymint_integrals/cusymint_integrals.dart';
import 'package:fluent_assertions/fluent_assertions.dart';
import 'package:test/test.dart';

import '../symbol_mock.dart';

void main() {
  test('toTex', () {
    final left = LeftSymbolMock();
    final right = RightSymbolMock();

    final multiplication = Multiplication(left, right);

    final result = multiplication.toTex();

    result.shouldBeEqualTo('left \\cdot right');
  });

  test('toString', () {
    final left = LeftSymbolMock();
    final right = RightSymbolMock();

    final multiplication = Multiplication(left, right);

    final result = multiplication.toString();

    result.shouldBeEqualTo('left * right');
  });

  test('toUtf', () {
    final left = LeftSymbolMock();
    final right = RightSymbolMock();

    final multiplication = Multiplication(left, right);

    final result = multiplication.toUtf();

    result.shouldBeEqualTo('left * right');
  });
}
