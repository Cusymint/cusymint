import 'package:cusymint_integrals/cusymint_integrals.dart';
import 'package:fluent_assertions/fluent_assertions.dart';
import 'package:test/test.dart';

import '../symbol_mock.dart';

void main() {
  test('toTex', () {
    final left = LeftSymbolMock();
    final right = RightSymbolMock();

    final substraction = Substraction(left, right);

    final result = substraction.toTex();

    result.shouldBeEqualTo('left - right');
  });

  test('toString', () {
    final left = LeftSymbolMock();
    final right = RightSymbolMock();

    final substraction = Substraction(left, right);

    final result = substraction.toString();

    result.shouldBeEqualTo('left - right');
  });

  test('toUtf', () {
    final left = LeftSymbolMock();
    final right = RightSymbolMock();

    final substraction = Substraction(left, right);

    final result = substraction.toUtf();

    result.shouldBeEqualTo('left - right');
  });
}
