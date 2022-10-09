import 'package:bloc_test/bloc_test.dart';
import 'package:cusymint_app/features/home/blocs/client_cubit.dart';
import 'package:cusymint_client_mock/cusymint_client_mock.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  final waitDuration = const Duration(milliseconds: 100);

  blocTest(
    'emits [] when nothing is added',
    build: () {
      final client = CusymintClientMock(
        fakeResponse: ResponseMockFactory.defaultResponse,
        delay: waitDuration,
      );

      final clientCubit = ClientCubit(client: client);

      return clientCubit;
    },
    expect: () => [],
  );

  blocTest(
    'emits ClientInitial after reset',
    build: () {
      final client = CusymintClientMock(
        fakeResponse: ResponseMockFactory.defaultResponse,
        delay: waitDuration,
      );

      final clientCubit = ClientCubit(client: client);

      return clientCubit;
    },
    act: (ClientCubit cubit) {
      cubit.reset();
    },
    expect: () => [isA<ClientInitial>()],
  );

  blocTest(
    'emits ClientLoading and ClientSuccess after successful solveIntegral',
    build: () {
      final client = CusymintClientMock(
        fakeResponse: ResponseMockFactory.defaultResponse,
        delay: waitDuration,
      );

      final clientCubit = ClientCubit(client: client);

      return clientCubit;
    },
    act: (ClientCubit cubit) async {
      await cubit.solveIntegral('x');
    },
    wait: waitDuration + waitDuration, // race condition
    expect: () => [
      isA<ClientLoading>(),
      isA<ClientSuccess>()
          .having(
            (cs) => cs.response.inputInTex,
            'Input in tex matches that produced by client',
            same(ResponseMockFactory.defaultResponse.inputInTex),
          )
          .having(
            (cs) => cs.response.inputInUtf,
            'Input in utf matches that produced by client',
            same(ResponseMockFactory.defaultResponse.inputInUtf),
          )
          .having(
            (cs) => cs.response.outputInTex,
            'Output in tex matches that produced by client',
            same(ResponseMockFactory.defaultResponse.outputInTex),
          )
          .having(
            (cs) => cs.response.outputInUtf,
            'Output in utf matches that produced by client',
            same(ResponseMockFactory.defaultResponse.outputInUtf),
          )
          .having(
            (cs) => cs.response.hasErrors,
            'No errors',
            isFalse,
          ),
    ],
  );

  blocTest(
    'emits ClientLoading and ClientFailure after client failure',
    build: () {
      final client = CusymintClientMock(
        fakeResponse: ResponseMockFactory.validationErrors,
        delay: waitDuration,
      );

      final clientCubit = ClientCubit(client: client);

      return clientCubit;
    },
    act: (ClientCubit cubit) async {
      await cubit.solveIntegral('x');
    },
    wait: waitDuration + waitDuration, // race condition
    expect: () => [
      isA<ClientLoading>(),
      isA<ClientFailure>().having(
        (cf) => cf.errors,
        'Has errors',
        isNotEmpty,
      ),
    ],
  );

  blocTest(
    'emits ClientLoading and ClientFailure on Client exception',
    build: () {
      final client = ExceptionThrowingClient(waitDuration: waitDuration);

      final clientCubit = ClientCubit(client: client);

      return clientCubit;
    },
    act: (ClientCubit cubit) async {
      await cubit.solveIntegral('x');
    },
    wait: waitDuration + waitDuration, // race condition
    expect: () => [
      isA<ClientLoading>(),
      isA<ClientFailure>().having(
        (cf) => cf.errors,
        'Has errors',
        isNotEmpty,
      ),
    ],
  );
}

class ExceptionThrowingClient extends Fake implements CusymintClient {
  ExceptionThrowingClient({required this.waitDuration});

  final Duration waitDuration;

  @override
  Future<Response> solveIntegral(Request request) async {
    await Future<void>.delayed(waitDuration);

    throw Exception('Fake exception');
  }
}
