import 'package:cusymint_client_mock/cusymint_client_mock.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

class ClientCubit extends Cubit<ClientState> {
  ClientCubit({required this.client}) : super(const ClientInitial());

  final CusymintClient client;

  Future<void> solveIntegral(String integralToBeSolved) async {
    final watch = Stopwatch()..start();

    final request = Request(integralToBeSolved);

    emit(ClientLoading(request: request));

    final response = await _getResponse(request);

    watch.stop();

    if (response.hasErrors) {
      emit(ClientFailure(
        request: request,
        errors: response.errors.map((e) => e.errorMessage).toList(),
      ));
      return;
    }

    final duration = watch.elapsed;

    emit(ClientSuccess(
      response: response,
      request: request,
      duration: duration,
    ));
  }

  Future<Response> _getResponse(Request request) async {
    try {
      final response = await client.solveIntegral(request);

      return response;
    } catch (e) {
      return Response(errors: [ResponseError(e.toString())]);
    }
  }

  void reset() {
    emit(const ClientInitial());
  }
}

abstract class ClientState {
  const ClientState();
}

class ClientInitial extends ClientState {
  const ClientInitial();
}

class ClientLoading extends ClientState {
  const ClientLoading({
    required this.request,
  });

  final Request request;
}

class ClientSuccess extends ClientState {
  ClientSuccess({
    required this.duration,
    required this.request,
    required this.response,
  }) {
    assert(response.hasErrors == false);
    assert(response.outputInUtf != null);
    assert(response.outputInTex != null);
    assert(response.inputInUtf != null);
    assert(response.inputInTex != null);
  }

  final Duration duration;
  final Request request;
  final Response response;
}

class ClientFailure extends ClientState {
  const ClientFailure({
    required this.request,
    this.errors = const [],
  });

  final Request request;
  final List<String> errors;
}
