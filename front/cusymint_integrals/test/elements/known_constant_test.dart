import 'package:cusymint_integrals/cusymint_integrals.dart';
import 'package:fluent_assertions/fluent_assertions.dart';
import 'package:test/test.dart';

void main() {
  group('Euler constant', () {
    test('toTex', () {
      final eulerConstant = EulerConstant();

      final result = eulerConstant.toTex();

      result.shouldBeEqualTo('e');
    });

    test('toString', () {
      final eulerConstant = EulerConstant();

      final result = eulerConstant.toString();

      result.shouldBeEqualTo('e');
    });

    test('toUtf', () {
      final eulerConstant = EulerConstant();

      final result = eulerConstant.toUtf();

      result.shouldBeEqualTo('e');
    });
  });

  group('Pi', () {
    test('toTex', () {
      final pi = PiConstant();

      final result = pi.toTex();

      result.shouldBeEqualTo('\\pi');
    });

    test('toString', () {
      final pi = PiConstant();

      final result = pi.toString();

      result.shouldBeEqualTo('π');
    });

    test('toUtf', () {
      final pi = PiConstant();

      final result = pi.toUtf();

      result.shouldBeEqualTo('π');
    });
  });
}
