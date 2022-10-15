import 'package:cusymint_client_interface/cusymint_client_interface.dart';

class CusymintClientMock implements CusymintClient {
  final Response fakeResponse;
  final Duration delay;

  @override
  Future<Response> solveIntegral(Request request) async {
    await Future.delayed(delay);
    return Future.value(fakeResponse);
  }

  CusymintClientMock({
    this.fakeResponse = const Response(),
    this.delay = const Duration(milliseconds: 800),
  });
}

class ResponseMockFactory {
  static Response get defaultResponse => Response(
        inputInUtf: '∫x^2 + sin(x) / cos(x) dx',
        inputInTex: '\\int x^2 + \\frac{\\sin(x)}{\\cos(x)} dx',
        outputInUtf: 'x^3/3 + log(cos(x)) + C',
        outputInTex: '\\frac{x^3}{3} + \\log{\\left(\\cos{x}\\right)} + C',
      );

  static Response get validationErrors => Response(
        errors: [
          const ResponseError('Could not parse "x^@"'),
        ],
      );
}
