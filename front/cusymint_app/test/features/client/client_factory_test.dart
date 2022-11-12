import 'package:cusymint_app/features/client/client_factory.dart';
import 'package:cusymint_client_json_rpc/cusymint_client_json_rpc.dart';
import 'package:cusymint_client_mock/cusymint_client_mock.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('isValid', () {
    final clientFactory = ClientFactory();

    final validUrls = <String>[
      'mock',
      'errors',
      'ws://localhost:8000/websocket',
      'ws://localhost/websocket/',
    ];

    for (final validUrl in validUrls) {
      test('valid url: $validUrl', () {
        expect(clientFactory.isUrlCorrect(validUrl), true);
      });
    }

    final invalidUrls = <String>[
      'http://locahost',
      'ws/localhost:8000/websocket',
      'mocks',
      'err',
      'localhost',
    ];

    for (final invalidUrl in invalidUrls) {
      test('invalid url: $invalidUrl', () {
        expect(clientFactory.isUrlCorrect(invalidUrl), false);
      });
    }
  });

  group('setUrl', () {
    test('Mock client', () {
      final factory = ClientFactory();

      factory.setUrl('mock');

      expect(
        factory.client,
        isA<CusymintClientMock>().having(
          (mock) => mock.fakeResponse.errors,
          'No errors',
          [],
        ).having(
          (mock) => mock.fakeResponse.outputInUtf,
          'Some output',
          isNotEmpty,
        ),
      );
    });

    test('Mock client with errors', () {
      final factory = ClientFactory();

      factory.setUrl('errors');

      expect(
        factory.client,
        isA<CusymintClientMock>()
            .having(
              (mock) => mock.fakeResponse.errors,
              'Some errors',
              isNotEmpty,
            )
            .having(
              (mock) => mock.fakeResponse.outputInUtf,
              'No output',
              isNull,
            ),
      );
    });

    test('JsonRpc client', () {
      final factory = ClientFactory();

      factory.setUrl('ws://localhost:8000/websocket');

      expect(
        factory.client,
        isA<CusymintClientJsonRpc>().having(
          (jsonRpc) => jsonRpc.uri,
          'Correct uri',
          Uri.parse('ws://localhost:8000/websocket'),
        ),
      );
    });
  });
}
