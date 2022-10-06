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
    expect: () => [isA<ClientLoading>(), isA<ClientSuccess>()],
  );
}
