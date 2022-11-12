import 'dart:async';

import 'package:cusymint_client_interface/cusymint_client_interface.dart';
import 'package:cusymint_client_json_rpc/src/cusymint_client_json_rpc.dart';
import 'package:json_rpc_2/json_rpc_2.dart';
import 'package:test/test.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

void main() {
  final uri = Uri.parse('ws://localhost:8000/websocket');

  test('Solve method returns input in Utf', () async {
    final client = CusymintClientJsonRpc(uri: uri);

    final request = Request('x^2');

    final response = await client.solveIntegral(request);

    expect(response.inputInUtf, 'x^2');
  }, skip: true, timeout: Timeout(Duration(days: 1)));

  test(
    'Test rpc.list',
    () async {
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
        ]),
      );
    },
    skip: true,
  );
}
