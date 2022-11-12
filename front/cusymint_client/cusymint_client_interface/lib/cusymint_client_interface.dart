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

  Response copyWith({
    String? inputInUtf,
    String? inputInTex,
    String? outputInUtf,
    String? outputInTex,
    List<ResponseError>? errors,
  }) {
    return Response(
      inputInUtf: inputInUtf ?? this.inputInUtf,
      inputInTex: inputInTex ?? this.inputInTex,
      outputInUtf: outputInUtf ?? this.outputInUtf,
      outputInTex: outputInTex ?? this.outputInTex,
      errors: errors ?? this.errors,
    );
  }

  bool get hasErrors => errors.isNotEmpty;
}

class ResponseError {
  final String errorMessage;

  const ResponseError(this.errorMessage);
}

abstract class CusymintClient {
  Future<Response> solveIntegral(Request request);
  Future<Response> interpretIntegral(Request request);
}
