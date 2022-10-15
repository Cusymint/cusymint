library cusymint_client_interface;

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

abstract class CusymintClient {
  Future<Response> solveIntegral(Request request);
}
