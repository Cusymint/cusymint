import 'package:cusymint_client_mock/cusymint_client_mock.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

class ClientCubit extends Cubit<ClientState> {
  ClientCubit({required this.client}) : super(const ClientInitial());

  final CusymintClient client;

  Future<void> solveIntegral(String integralToBeSolved) async {
    final request = Request(integralToBeSolved);

    emit(ClientLoading(request: request));

    try {
      final response = await client.solveIntegral(request);

      if (response.hasErrors) {
        emit(ClientFailure(
          request: request,
          errors: response.errors.map((e) => e.errorMessage).toList(),
        ));
        return;
      }

      emit(ClientSuccess(
        request: request,
        response: response,
      ));
    } catch (e) {
      emit(ClientFailure(
        request: request,
        errors: [e.toString()],
      ));
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
  const ClientSuccess({
    required this.request,
    required this.response,
  });

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
