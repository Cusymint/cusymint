import 'dart:async';

import 'package:cusymint_client/cusymint_client.dart';
import 'package:json_rpc_2/json_rpc_2.dart';
import 'package:test/test.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

void main() {
  group('json_rpc integration tests', () {
    final uri = Uri.parse('ws://localhost:8000/websocket');

    test('Correct solve method returns result', () async {
      final client = CusymintClientJsonRpc(uri: uri);

      final request = Request('x^2');

      final response = await client.solveIntegral(request);

      expect(response.inputInUtf, isNotNull);
      expect(response.inputInTex, isNotNull);

      expect(response.outputInUtf, isNotNull);
      expect(response.outputInTex, isNotNull);

      expect(response.steps, isNull);

      expect(response.errors, isEmpty);
    });

    test('Incorrect solve method returns UnexpectedToken error', () async {
      final client = CusymintClientJsonRpc(uri: uri);

      final request = Request('x^2 +');

      final response = await client.solveIntegral(request);

      expect(response.inputInUtf, isNull);
      expect(response.inputInTex, isNull);

      expect(response.outputInUtf, isNull);
      expect(response.outputInTex, isNull);

      expect(response.steps, isNull);

      expect(response.errors, isNotEmpty);
      expect(
        response.errors,
        allOf(
          isNotEmpty,
          contains(
            isA<UnexpectedEndOfInputError>(),
          ),
        ),
      );
    });

    test('Correct solve with steps method returns result', () async {
      final client = CusymintClientJsonRpc(uri: uri);

      final request = Request('x^2');

      final response = await client.solveIntegralWithSteps(request);

      expect(response.inputInUtf, isNotNull);
      expect(response.inputInTex, isNotNull);

      expect(response.outputInUtf, isNotNull);
      expect(response.outputInTex, isNotNull);

      expect(response.steps, isNotNull);

      expect(response.errors, isEmpty);
    });

    test('Incorrect solve with steps method returns UnexpectedToken error',
        () async {
      final client = CusymintClientJsonRpc(uri: uri);

      final request = Request('x^2 +');

      final response = await client.solveIntegralWithSteps(request);

      expect(response.inputInUtf, isNull);
      expect(response.inputInTex, isNull);

      expect(response.outputInUtf, isNull);
      expect(response.outputInTex, isNull);

      expect(response.steps, isNull);

      expect(response.errors, isNotEmpty);
      expect(
        response.errors,
        allOf(
          isNotEmpty,
          contains(
            isA<UnexpectedEndOfInputError>(),
          ),
        ),
      );
    });

    test('Incorrect interpret method returns UnexpectedToken error', () async {
      final client = CusymintClientJsonRpc(uri: uri);

      final request = Request('x^2 +');

      final response = await client.interpretIntegral(request);

      expect(response.inputInUtf, isNull);
      expect(response.inputInTex, isNull);

      expect(response.outputInUtf, isNull);
      expect(response.outputInTex, isNull);

      expect(response.steps, isNull);

      expect(response.errors, isNotEmpty);
      expect(
        response.errors,
        allOf(
          isNotEmpty,
          contains(
            isA<UnexpectedEndOfInputError>(),
          ),
        ),
      );
    });

    test('Interpret method returns input in Utf and Tex', () async {
      final client = CusymintClientJsonRpc(uri: uri);

      final request = Request('x^2');

      final response = await client.interpretIntegral(request);

      expect(response.inputInUtf, isNotNull);
      expect(response.inputInTex, isNotNull);

      expect(response.outputInUtf, isNull);
      expect(response.outputInTex, isNull);

      expect(response.steps, isNull);

      expect(response.errors, isEmpty);
    });

    test('Interpret method returns errors', () async {
      final client = CusymintClientJsonRpc(uri: uri);

      final request = Request('unparsable');

      final response = await client.interpretIntegral(request);

      expect(response.inputInUtf, isNull);
      expect(response.inputInTex, isNull);

      expect(response.outputInUtf, isNull);
      expect(response.outputInTex, isNull);

      expect(response.steps, isNull);

      expect(response.errors, isNotEmpty);
    });

    test('Test rpc.list', () async {
      var socket = WebSocketChannel.connect(uri);
      var client = Client(socket.cast<String>());

      unawaited(client.listen());

      final response = await client.sendRequest(
        'rpc.list',
      );

      final methodsList = response as List;

      expect(
        methodsList,
        allOf([
          contains('rpc.list'),
          contains('solve'),
          contains('interpret'),
          contains('solve_with_steps'),
        ]),
      );
    });
  });
}
