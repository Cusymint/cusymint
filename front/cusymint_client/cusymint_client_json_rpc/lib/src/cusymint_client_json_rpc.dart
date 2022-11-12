import 'dart:async';

import 'package:json_rpc_2/json_rpc_2.dart';
import 'package:cusymint_client_interface/cusymint_client_interface.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

class CusymintClientJsonRpc implements CusymintClient {
  CusymintClientJsonRpc({
    required this.uri,
  });

  final Uri uri;
  static const String _solveMethodName = 'solve';
  static const String _interpretMethodName = 'interpret';

  @override
  Future<Response> solveIntegral(Request request) async {
    var socket = WebSocketChannel.connect(uri);
    var client = Client(socket.cast<String>());

    unawaited(client.listen());

    try {
      // this should return result, but for some reason it doesn't
      final result = await client.sendRequest(
        _solveMethodName,
        {'input': request.integralToBeSolved},
      );

      final errors = result['errors'] != null
          ? (result['errors'] as List)
              .map((e) => ResponseError(e['errorMessage']))
              .toList()
          : List<ResponseError>.empty();

      return Response(
        inputInUtf: result['inputInUtf'],
        inputInTex: result['inputInTex'],
        outputInUtf: result['outputInUtf'],
        outputInTex: result['outputInTex'],
        errors: errors,
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
