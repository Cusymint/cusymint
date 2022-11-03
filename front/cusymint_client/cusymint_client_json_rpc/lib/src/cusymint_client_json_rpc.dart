import 'dart:async';

import 'package:json_rpc_2/json_rpc_2.dart';
import 'package:cusymint_client_interface/cusymint_client_interface.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

class CusymintClientJsonRpc implements CusymintClient {
  CusymintClientJsonRpc({
    required this.url,
  });

  final String url;
  static const String _solveMethodName = 'solve';
  static const String _interpretMethodName = 'interpret';

  @override
  Future<Response> solveIntegral(Request request) async {
    var socket = WebSocketChannel.connect(Uri.parse(url));
    var client = Client(socket.cast<String>());

    unawaited(client.listen());

    try {
      var result = await client.sendRequest(
        _solveMethodName,
        {'input': request.integralToBeSolved},
      );
      return Response(
        inputInUtf: result['inputInUtf'],
        inputInTex: result['inputInTex'],
        outputInUtf: result['outputInUtf'],
        outputInTex: result['outputInTex'],
        errors: (result['errors'] as List)
            .map((e) => ResponseError(e['errorMessage']))
            .toList(),
      );
    } on RpcException catch (e) {
      return Response(
        errors: [ResponseError(e.message)],
      );
    } finally {
      client.close();
    }
  }
}
