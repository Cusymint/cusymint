import 'dart:async';

import 'package:json_rpc_2/json_rpc_2.dart' as json_rpc;
import 'package:web_socket_channel/web_socket_channel.dart';

import 'interface.dart';

// TODO: reuse channel
class CusymintClientJsonRpc implements CusymintClient {
  CusymintClientJsonRpc({
    required this.uri,
  });

  final Uri uri;
  static const String _solveMethodName = 'solve';
  static const String _interpretMethodName = 'interpret';

  @override
  Future<Response> solveIntegral(Request request) async {
    return await _handleRequest(_solveMethodName, request);
  }

  @override
  Future<Response> interpretIntegral(Request request) async {
    return await _handleRequest(_interpretMethodName, request);
  }

  Future<Response> _handleRequest(String methodName, Request request) async {
    var socket = WebSocketChannel.connect(uri);
    var client = json_rpc.Client(socket.cast<String>());

    unawaited(client.listen());

    try {
      // this should return result, but for some reason it doesn't
      final result = await client.sendRequest(
        methodName,
        {'input': request.integralToBeSolved},
      );

      final errors = result['errors'] != null
          ? (result['errors'] as List).map((e) => ResponseError(e)).toList()
          : List<ResponseError>.empty();

      return Response(
        inputInUtf: result['inputInUtf'],
        inputInTex: result['inputInTex'],
        outputInUtf: result['outputInUtf'],
        outputInTex: result['outputInTex'],
        errors: errors,
      );
    } on json_rpc.RpcException catch (e) {
      return Response(
        errors: [ResponseError(e.message)],
      );
    } finally {
      client.close();
    }
  }
}
