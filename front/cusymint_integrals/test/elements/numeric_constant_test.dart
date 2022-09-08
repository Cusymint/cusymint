import 'package:cusymint_integrals/cusymint_integrals.dart';
import 'package:fluent_assertions/fluent_assertions.dart';
import 'package:test/test.dart';

void main() {
  var inputsToExpected = {
    0.0: '0',
    1.0: '1',
    2.0: '2',
    3.14: '3.14',
    -123.456: '-123.456',
  };

  inputsToExpected.forEach((input, expected) {
    test('toTex $input -> $expected', () {
      final constant = NumericConstant(input);

      final result = constant.toTex();

      result.shouldBeEqualTo(expected);
    });

    test('toString $input -> $expected', () {
      final constant = NumericConstant(input);

      final result = constant.toString();

      result.shouldBeEqualTo(expected);
    });

    test('toUtf $input -> $expected', () {
      final constant = NumericConstant(input);

      final result = constant.toUtf();

      result.shouldBeEqualTo(expected);
    });
  });
}
