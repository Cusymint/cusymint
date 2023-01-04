import 'package:cusymint_client/cusymint_client.dart';
import 'package:test/test.dart';

void main() {
  group('json_rpc error parsing', () {
    group('Unexpected Tokens', () {
      final correctErrorMessages = <String, String>{
        'Unexpected token: xdx': 'xdx',
        'Unexpected token: \\': '\\',
        'Unexpected token: \\int': '\\int',
        'Unexpected token: _13': '_13',
      };

      correctErrorMessages.forEach((errorMessage, expectedToken) {
        test('Correct error message: $errorMessage', () {
          final error = CusymintClientJsonRpc.parseError(errorMessage);

          expect(
            error,
            isA<UnexpectedTokenError>().having(
              (err) => err.token,
              'Token',
              expectedToken,
            ),
          );
        });
      });

      final endOfLineErrorMessage = 'Unexpected token: <end>';
      test('UnexpectedEndOfInput', () {
        final error = CusymintClientJsonRpc.parseError(endOfLineErrorMessage);
        expect(error, isA<UnexpectedEndOfInputError>());
      });

      final incorrectErrorMessages = [
        'SZSZ: xdx',
        'Unexpected Token: \\',
        '',
      ];

      for (var errorMessage in incorrectErrorMessages) {
        test('Incorrect error message: $errorMessage', () {
          final error = CusymintClientJsonRpc.parseError(errorMessage);

          expect(
            error,
            isNot(isA<UnexpectedTokenError>()),
          );
        });
      }
    });

    test('Internal error', () {
      final errorMessage = 'Internal error';

      final error = CusymintClientJsonRpc.parseError(errorMessage);

      expect(
        error,
        isA<InternalError>(),
      );
    });

    test('No solution found', () {
      final errorMessage = 'No solution found';

      final error = CusymintClientJsonRpc.parseError(errorMessage);

      expect(
        error,
        isA<NoSolutionFoundError>(),
      );
    });
  });
}
