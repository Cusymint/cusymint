import 'package:cusymint_app/features/client/client_factory.dart';
import 'package:cusymint_client_mock/cusymint_client_mock.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group(
    'ClientFactory tests',
    () {
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
    },
  );
}
