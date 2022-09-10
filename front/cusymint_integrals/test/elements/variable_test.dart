import 'package:cusymint_integrals/cusymint_integrals.dart';
import 'package:fluent_assertions/fluent_assertions.dart';
import 'package:test/test.dart';

void main() {
  test('toTex', () {
    final variable = Variable();

    final result = variable.toTex();

    result.shouldBe('x');
  });

  test('toString', () {
    final variable = Variable();

    final result = variable.toString();

    result.shouldBe('x');
  });

  test('toUtf', () {
    final variable = Variable();

    final result = variable.toUtf();

    result.shouldBe('x');
  });
}
