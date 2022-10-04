library cusymint_client_interface;

class Request {
  final String integralToBeSolved;

  Request(this.integralToBeSolved);
}

class Response {
  final String? inputInUtf;
  final String? inputInTex;

  final String? outputInUtf;
  final String? outputInTex;

  final List<ResponseErrors> errors;

  Response({
    this.inputInUtf,
    this.inputInTex,
    this.outputInUtf,
    this.outputInTex,
    this.errors = const [],
  });

  bool get hasErrors => errors.isNotEmpty;
}

class ResponseErrors {
  final String errorMessage;

  ResponseErrors(this.errorMessage);
}

abstract class CusymintClient {
  Future<Response> solveIntegral(Request request);
}
