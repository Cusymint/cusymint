import 'dart:math';

import 'package:cusymint_app/features/client/client_factory.dart';
import 'package:cusymint_client_json_rpc/cusymint_client_json_rpc.dart';
import 'package:cusymint_client_mock/cusymint_client_mock.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

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

  TypeMatcher<CusymintClientMock> isAClientMockWithErrors() =>
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
          );

  TypeMatcher<CusymintClientMock> isAClientMockWithOutput() =>
      isA<CusymintClientMock>().having(
        (mock) => mock.fakeResponse.errors,
        'No errors',
        [],
      ).having(
        (mock) => mock.fakeResponse.outputInUtf,
        'Some output',
        isNotEmpty,
      );

  group('setUrl', () {
    test('Mock client', () {
      final factory = ClientFactory();

      factory.setUrl('mock');

      expect(factory.client, isAClientMockWithOutput());
    });

    test('Mock client with errors', () {
      final factory = ClientFactory();

      factory.setUrl('errors');

      expect(factory.client, isAClientMockWithErrors());
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

    test('Stores url using SharedPreferences', () async {
      SharedPreferences.setMockInitialValues({});
      final prefs = await SharedPreferences.getInstance();

      final factorySave = ClientFactory(sharedPreferences: prefs);

      expect(factorySave.client, isNot(isAClientMockWithErrors()));

      const url = 'errors';
      factorySave.setUrl(url);

      expect(factorySave.client, isAClientMockWithErrors());

      final factoryLoad = ClientFactory(sharedPreferences: prefs);

      expect(factoryLoad.client, isAClientMockWithErrors());

      final factoryWithoutPrefs = ClientFactory();

      expect(factoryWithoutPrefs.client, isNot(isAClientMockWithErrors()));
    });
  });
}
