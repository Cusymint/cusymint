import 'interface.dart';

class CusymintClientMock implements CusymintClient {
  final Response fakeResponse;
  final Duration solveDelay;
  final Duration interpretDelay;

  @override
  Future<Response> solveIntegral(Request request) async {
    await Future.delayed(solveDelay);
    return Future.value(fakeResponse);
  }

  @override
  Future<Response> solveIntegralWithSteps(Request request) {
    // TODO: implement solveIntegralWithSteps
    throw UnimplementedError();
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
  );

  static const Response validationErrors = Response(
    errors: [
      ResponseError('Could not parse "x^@"'),
    ],
  );
}
