import 'package:cusymint_client_json_rpc/cusymint_client_json_rpc.dart';
import 'package:cusymint_client_mock/cusymint_client_mock.dart';
import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:provider/provider.dart';

class ClientFactory {
  ClientFactory();

  Uri get uri => _uri;
  Uri _uri = Uri.parse(_defaultUri);

  CusymintClient get client => _client;
  CusymintClient _client = CusymintClientMock();

  static const _defaultUri = 'ws://localhost:8000/websocket';

  void setUrl(String url) {
    _uri = Uri.parse(url);

    if (url == 'mock') {
      _client = CusymintClientMock(
        fakeResponse: ResponseMockFactory.defaultResponse,
      );
      return;
    }

    if (url == 'errors') {
      _client = CusymintClientMock(
        fakeResponse: ResponseMockFactory.validationErrors,
      );
      return;
    }

    if (_uri.scheme == 'ws') {
      _client = CusymintClientJsonRpc(uri: _uri);
      return;
    }

    throw Exception('Unsupported scheme: ${_uri.scheme}');
  }

  static ClientFactory of(BuildContext context) =>
      Provider.of<ClientFactory>(context, listen: false);
}

extension on CusymintClient {
  CusymintClient of(BuildContext context) {
    return Provider.of<ClientFactory>(context, listen: false).client;
  }
}
