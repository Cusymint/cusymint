import 'interface.dart';

class CusymintClientMock implements CusymintClient {
  final Response fakeResponse;
  final Duration solveDelay;
  final Duration interpretDelay;

  @override
  Future<Response> solveIntegral(Request request) async {
    await Future.delayed(solveDelay);
    final responseWithoutHistory = Response(
      inputInUtf: fakeResponse.inputInUtf,
      inputInTex: fakeResponse.inputInTex,
      outputInUtf: fakeResponse.outputInUtf,
      outputInTex: fakeResponse.outputInTex,
      errors: fakeResponse.errors,
    );
    return Future.value(responseWithoutHistory);
  }

  @override
  Future<Response> solveIntegralWithSteps(Request request) async {
    await Future.delayed(solveDelay);
    return Future.value(fakeResponse);
  }

  @override
  Future<Response> interpretIntegral(Request request) async {
    await Future.delayed(interpretDelay);
    final responseWithoutOutput = Response(
      inputInUtf: fakeResponse.inputInUtf,
      inputInTex: fakeResponse.inputInTex,
      errors: fakeResponse.errors,
    );

    return Future.value(responseWithoutOutput);
  }

  const CusymintClientMock({
    this.fakeResponse = const Response(),
    this.solveDelay = const Duration(milliseconds: 500),
    this.interpretDelay = const Duration(milliseconds: 30),
  });
}

class ResponseMockFactory {
  static const Response defaultResponse = Response(
    inputInUtf: '∫x^2 + sin(x) / cos(x) dx',
    inputInTex: '\\int x^2 + \\frac{\\sin(x)}{\\cos(x)} dx',
    outputInUtf: 'x^3/3 + log(cos(x)) + C',
    outputInTex: '\\frac{x^3}{3} + \\log{\\left(\\cos{x}\\right)} + C',
    steps: '\\quad \\text{[[simplify]]}:\\newline'
        '=\\qquad \\int \\frac{ \\sin\\left(x\\right) }'
        '{ \\cos\\left(x\\right) }+x^{ 2 }\\text{d} x\\newline'
        '=\\qquad \\int \\frac{ \\sin\\left(x\\right) }'
        '{ \\cos\\left(x\\right) }\\text{d} '
        'x+\\int x^{ 2 }\\text{d} x\\newline\\quad'
        ' \\text{[[solveIntegral]]:} \\int x^{ 2 } \\text{d} x '
        '= \\frac{ x^{ 2+1 } }{ 2+1 }'
        ' + C:\\newline=\\qquad \\int \\frac{'
        ' \\sin\\left(x\\right) }{ \\cos\\left(x\\right) }'
        '\\text{d} x+0.333333x^{ 3 }\\newline\\quad \\text{[[substitute]]}'
        '\\: u=\\cos\\left(x\\right), \\text{d}'
        ' u=-\\sin\\left(x\\right) \\text{d} x:\\newline=\\qquad \\int'
        ' -\\frac{ 1 }{ u }\\text{d} u_{ u = \\cos\\left(x\\right)'
        ' }+0.333333x^{ 3 }\\newline=\\qquad -\\int'
        ' \\frac{ 1 }{ u }\\text{d} '
        'u_{ u = \\cos\\left(x\\right) }+0.333333x^{ 3 }\\newline\\quad '
        '\\text{[[solveIntegral]]:} \\int \\frac{ 1 }{ u } '
        '\\text{d} u = \\ln\\left(u\\right) + C:\\newline=\\qquad '
        '0.333333x^{ 3 }-\\ln\\left(\\cos\\left(x\\right)\\right)',
  );

  static const Response validationErrors = Response(
    errors: [
      ResponseError('Could not parse "x^@"'),
    ],
  );
}
