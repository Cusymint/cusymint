import 'package:cusymint_client_interface/cusymint_client_interface.dart';
import 'package:cusymint_client_json_rpc/src/cusymint_client_json_rpc.dart';
import 'package:test/test.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

void main() {
  test('Solve method returns input in Utf', () async {
    final client = CusymintClientJsonRpc(url: 'ws://localhost:8000/websocket');

    final request = Request('x^2');

    final response = await client.solveIntegral(request);

    expect(response.inputInUtf, 'x^2');
  });

  test('Test websockets', () async {
    final socket = WebSocketChannel.connect(
      Uri.parse('ws://localhost:8000/websocket'),
    );

    socket.stream.listen((event) {
      assert(event is String);
      print(event);
    });

    socket.sink.add(
      {
        'jsonrpc': '2.0',
        'id': 1,
        'method': 'solve',
      },
    );

    await Future.delayed(Duration(seconds: 20));
  });
}
