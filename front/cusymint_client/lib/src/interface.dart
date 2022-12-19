class Request {
  final String integralToBeSolved;

  const Request(this.integralToBeSolved);
}

class Response {
  final String? inputInUtf;
  final String? inputInTex;

  final String? outputInUtf;
  final String? outputInTex;

  final List<ResponseError> errors;

  const Response({
    this.inputInUtf,
    this.inputInTex,
    this.outputInUtf,
    this.outputInTex,
    this.errors = const [],
  });

  bool get hasErrors => errors.isNotEmpty;
}

class ResponseError {
  final String errorMessage;

  const ResponseError(this.errorMessage);
}

class InternalError extends ResponseError {
  const InternalError({String errorMessage = "Internal error"})
      : super(errorMessage);
}

class NoSolutionFoundError extends ResponseError {
  const NoSolutionFoundError({String errorMessage = "No solution found"})
      : super(errorMessage);
}

class UnexpectedTokenError extends ResponseError {
  const UnexpectedTokenError({
    String errorMessage = "Unexpected token",
    required this.token,
  }) : super(errorMessage);

  final String token;
}

class UnexpectedEndOfInputError extends ResponseError {
  const UnexpectedEndOfInputError({
    String errorMessage = "Unexpected end of input",
  }) : super(errorMessage);
}

abstract class CusymintClient {
  Future<Response> solveIntegral(Request request);
  Future<Response> interpretIntegral(Request request);
}
