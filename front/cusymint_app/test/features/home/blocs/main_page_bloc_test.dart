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
    wait: solveDelay + solveDelay,
    expect: () => [],
  );

  blocTest(
    'emits loading state and state with solution when SolveRequested',
    build: () => _createBloc(clientMock),
    act: (MainPageBloc bloc) => bloc.add(const SolveRequested('x')),
    wait: solveDelay + solveDelay,
    expect: () => [
      _isMainPageStateLoadingWithInput('x')
          .having((s) => s.errors, 'errors', isEmpty),
      isA<MainPageState>()
          .having(
            (state) => state.userInput,
            'correct input',
            equals('x'),
          )
          ._havingInputInTex(correctResponse.inputInTex)
          ._havingInputInUtf(correctResponse.inputInUtf)
          ._havingOutputInTex(correctResponse.outputInTex)
          ._havingOutputInUtf(correctResponse.outputInUtf)
          ._notHavingErrors(),
    ],
  );

  blocTest(
    'emits loading state and state with errors after client failure',
    build: () => _createBloc(clientFailuresMock),
    act: (MainPageBloc bloc) => bloc.add(const SolveRequested('x')),
    wait: solveDelay + solveDelay,
    expect: () => [
      _isMainPageStateLoadingWithInput('x'),
      isA<MainPageState>()
          .having(
            (state) => state.userInput,
            'correct user input',
            equals('x'),
          )
          ._havingErrors(),
    ],
  );

  blocTest(
    'emits loading state and error state on client exception',
    build: () => _createBloc(
      ExceptionThrowingClient(waitDuration: solveDelay),
    ),
    act: (MainPageBloc bloc) => bloc.add(const SolveRequested('x')),
    wait: solveDelay + solveDelay,
    expect: () => [
      _isMainPageStateLoadingWithInput('x'),
      isA<MainPageState>()
          .having(
            (state) => state.userInput,
            'correct user input',
            equals('x'),
          )
          ._havingErrors(),
    ],
  );

  blocTest(
    'emits loading and interpreted state when InputChanged',
    build: () => _createBloc(clientMock),
    wait: interpretDelay * 8,
    act: (MainPageBloc bloc) => bloc.add(const InputChanged('x')),
    expect: () => [
      _isMainPageStateLoadingWithInput('x')
          ._notHavingInput()
          ._notHavingOutput()
          ._notHavingErrors(),
      isA<MainPageState>()
          .having(
            (state) => state.userInput,
            'correct user input',
            equals('x'),
          )
          .having((state) => state.isLoading, 'loaded', isFalse)
          .having((state) => state.outputInTex, 'output', isNull)
          .having((state) => state.outputInUtf, 'output', isNull)
          ._havingInputInTex(correctResponse.inputInTex)
          ._havingInputInUtf(correctResponse.inputInUtf),
    ],
  );

  blocTest(
    'emits loading state and state with solution and history '
    'when StepsRequested',
    build: () => _createBloc(clientMock),
    act: (MainPageBloc bloc) => bloc.add(const StepsRequested('x')),
    wait: solveDelay + solveDelay,
    expect: () => [
      _isMainPageStateLoadingWithInput('x')
          .having((s) => s.errors, 'errors', isEmpty),
      isA<MainPageState>()
          .having(
            (state) => state.userInput,
            'correct input',
            equals('x'),
          )
          ._havingInputInTex(correctResponse.inputInTex)
          ._havingInputInUtf(correctResponse.inputInUtf)
          ._havingOutputInTex(correctResponse.outputInTex)
          ._havingOutputInUtf(correctResponse.outputInUtf)
          ._havingSteps(correctResponse.steps)
          ._notHavingErrors(),
    ],
  );

  blocTest(
    'emits loading state and error state on client exception'
    ' when StepsRequested',
    build: () => _createBloc(
      ExceptionThrowingClient(waitDuration: solveDelay),
    ),
    act: (MainPageBloc bloc) => bloc.add(const StepsRequested('x')),
    wait: solveDelay + solveDelay,
    expect: () => [
      _isMainPageStateLoadingWithInput('x'),
      isA<MainPageState>()
          .having(
            (state) => state.userInput,
            'correct user input',
            equals('x'),
          )
          ._havingErrors(),
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

TypeMatcher<MainPageState> _isMainPageStateLoadingWithInput(String input) =>
    isA<MainPageState>()
        .having(
          (state) => state.userInput,
          'correct user input',
          equals(input),
        )
        .having((state) => state.isLoading, 'loading', isTrue);

extension on TypeMatcher<MainPageState> {
  TypeMatcher<MainPageState> _havingInputInTex(String? input) =>
      having((state) => state.inputInTex, 'input', equals(input));

  TypeMatcher<MainPageState> _havingInputInUtf(String? input) =>
      having((state) => state.inputInUtf, 'input', equals(input));

  TypeMatcher<MainPageState> _havingOutputInTex(String? output) =>
      having((state) => state.outputInTex, 'output', equals(output));

  TypeMatcher<MainPageState> _havingOutputInUtf(String? output) =>
      having((state) => state.outputInUtf, 'output', equals(output));

  TypeMatcher<MainPageState> _havingSteps(String? steps) =>
      having((state) => state.steps, 'history', equals(steps));

  TypeMatcher<MainPageState> _havingErrors() =>
      having((state) => state.errors, 'errors', isNotEmpty);

  TypeMatcher<MainPageState> _notHavingErrors() =>
      having((state) => state.errors, 'errors', isEmpty);

  TypeMatcher<MainPageState> _notHavingInput() =>
      having((state) => state.inputInTex, 'inputInTex', isNull)
          .having((state) => state.inputInUtf, 'inputInUtf', isNull);

  TypeMatcher<MainPageState> _notHavingOutput() =>
      having((state) => state.outputInTex, 'outputInTex', isNull)
          .having((state) => state.outputInUtf, 'outputInUtf', isNull);
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
