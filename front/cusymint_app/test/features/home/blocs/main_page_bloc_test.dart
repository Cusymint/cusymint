import 'package:bloc_test/bloc_test.dart';
import 'package:cusymint_app/features/client/client_factory.dart';
import 'package:cusymint_app/features/home/blocs/main_page_bloc.dart';
import 'package:cusymint_client/cusymint_client.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mockito/mockito.dart';

void main() {
  const solveDelay = Duration(milliseconds: 400);
  const interpretDelay = Duration(milliseconds: 50);
  const correctResponse = ResponseMockFactory.defaultResponse;
  const clientMock = CusymintClientMock(
    solveDelay: solveDelay,
    interpretDelay: interpretDelay,
    fakeResponse: correctResponse,
  );
  const clientFailuresMock = CusymintClientMock(
    solveDelay: solveDelay,
    interpretDelay: interpretDelay,
    fakeResponse: ResponseMockFactory.validationErrors,
  );

  blocTest(
    'emits [] when nothing is added',
    build: () => _createBloc(clientMock),
    expect: () => [],
  );

  blocTest(
    'emits SolvingState and SolvedState when SolveRequested',
    build: () => _createBloc(clientMock),
    act: (MainPageBloc bloc) => bloc.add(const SolveRequested('x')),
    wait: solveDelay + solveDelay,
    expect: () => [
      isA<SolvingState>().having(
        (state) => state.userInput,
        'correct user input',
        equals('x'),
      ),
      isA<SolvedState>()
          .having(
            (state) => state.userInput,
            'correct result',
            equals('x'),
          )
          .having(
            (state) => state.inputInTex,
            'correct input',
            equals(correctResponse.inputInTex),
          )
          .having(
            (state) => state.inputInUtf,
            'correct input',
            equals(correctResponse.inputInUtf),
          )
          .having(
            (state) => state.outputInTex,
            'correct output',
            equals(correctResponse.outputInTex),
          )
          .having(
            (state) => state.outputInUtf,
            'correct output',
            equals(correctResponse.outputInUtf),
          ),
    ],
  );

  blocTest(
    'emits SolvingState and SolveErrorState after client failure',
    build: () => _createBloc(clientFailuresMock),
    act: (MainPageBloc bloc) => bloc.add(const SolveRequested('x')),
    wait: solveDelay + solveDelay,
    expect: () => [
      isA<SolvingState>().having(
        (state) => state.userInput,
        'correct user input',
        equals('x'),
      ),
      isA<SolveErrorState>()
          .having(
            (state) => state.userInput,
            'correct user input',
            equals('x'),
          )
          .having(
            (state) => state.errors,
            'some errors',
            isNotEmpty,
          ),
    ],
  );

  blocTest(
    'emits SolvingState and SolveErrorState on client exception',
    build: () => _createBloc(
      ExceptionThrowingClient(waitDuration: solveDelay),
    ),
    act: (MainPageBloc bloc) => bloc.add(const SolveRequested('x')),
    wait: solveDelay + solveDelay,
    expect: () => [
      isA<SolvingState>().having(
        (state) => state.userInput,
        'correct user input',
        equals('x'),
      ),
      isA<SolveErrorState>()
          .having(
            (state) => state.userInput,
            'correct user input',
            equals('x'),
          )
          .having(
            (state) => state.errors,
            'some errors',
            isNotEmpty,
          ),
    ],
  );

  blocTest(
    'emits InterpretingState and InterpretedState when InputChanged',
    build: () => _createBloc(clientMock),
    wait: interpretDelay * 8,
    act: (MainPageBloc bloc) => bloc.add(const InputChanged('x')),
    expect: () => [
      isA<InterpretingState>().having(
        (state) => state.userInput,
        'correct user input',
        equals('x'),
      ),
      isA<InterpretedState>()
          .having(
            (state) => state.userInput,
            'correct result',
            equals('x'),
          )
          .having(
            (state) => state.inputInTex,
            'correct input',
            equals(correctResponse.inputInTex),
          )
          .having(
            (state) => state.inputInUtf,
            'correct input',
            equals(correctResponse.inputInUtf),
          ),
    ],
  );

  test('ClientFactoryMock returns specified client', () {
    const client1 = clientFailuresMock;
    final factory1 = ClientFactoryMock(client1);
    expect(factory1.client, equals(client1));

    const client2 = clientMock;
    final factory2 = ClientFactoryMock(client2);
    expect(factory2.client, equals(client2));
  });
}

MainPageBloc _createBloc(CusymintClient client) {
  final clientFactory = ClientFactoryMock(client);
  final bloc = MainPageBloc(clientFactory: clientFactory);
  return bloc;
}

class ClientFactoryMock extends Mock implements ClientFactory {
  ClientFactoryMock(CusymintClient client) : _client = client;

  final CusymintClient _client;

  @override
  CusymintClient get client => _client;
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
