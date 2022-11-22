import 'package:cusymint_app/features/client/client_factory.dart';
import 'package:cusymint_client/cusymint_client.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

class ClientCubit extends Cubit<ClientState> {
  ClientCubit({required this.clientFactory}) : super(const ClientInitial());

  final ClientFactory clientFactory;

  Future<void> solveIntegral(String integralToBeSolved) async {
    final watch = Stopwatch()..start();

    final request = Request(integralToBeSolved);

    emit(const ClientLoading());

    final response = await _getResponse(request);

    watch.stop();

    if (response.hasErrors) {
      emit(ClientFailure(
        errors: response.errors.map((e) => e.errorMessage).toList(),
      ));
      return;
    }

    final duration = watch.elapsed;

    emit(ClientSuccess(
      inputInTex: response.inputInTex!,
      inputInUtf: response.inputInUtf!,
      outputInTex: response.outputInTex!,
      outputInUtf: response.outputInUtf!,
      duration: duration,
    ));
  }

  Future<Response> _getResponse(Request request) async {
    try {
      final client = clientFactory.client;

      final response = await client.solveIntegral(request);

      if (response.inputInTex == null ||
          response.inputInUtf == null ||
          response.outputInTex == null ||
          response.outputInUtf == null) {
        throw ArgumentError('Response is missing some fields');
      }

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
  const ClientLoading();
}

class ClientSuccess extends ClientState {
  const ClientSuccess({
    required this.inputInUtf,
    required this.inputInTex,
    required this.outputInUtf,
    required this.outputInTex,
    required this.duration,
  });

  final String inputInUtf;
  final String inputInTex;
  final String outputInUtf;
  final String outputInTex;

  final Duration duration;
}

class ClientFailure extends ClientState {
  const ClientFailure({
    this.errors = const [],
  });

  final List<String> errors;
}
