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

  final _clientMatchers = <_ClientMatcher>[
    _ClientMatcher(
      match: (uri) => uri.host == '' && uri.path == 'mock',
      create: (_) => CusymintClientMock(
        fakeResponse: ResponseMockFactory.defaultResponse,
      ),
    ),
    _ClientMatcher(
      match: (uri) => uri.host == '' && uri.path == 'errors',
      create: (_) => CusymintClientMock(
        fakeResponse: ResponseMockFactory.validationErrors,
      ),
    ),
    _ClientMatcher(
      match: (uri) => uri.scheme == 'ws',
      create: (uri) => CusymintClientJsonRpc(uri: uri),
    ),
  ];

  bool isUrlCorrect(String url) {
    for (final matcher in _clientMatchers) {
      if (matcher.match(Uri.parse(url))) {
        return true;
      }
    }

    return false;
  }

  void setUrl(String url) {
    _uri = Uri.parse(url);

    for (final matcher in _clientMatchers) {
      if (matcher.match(_uri)) {
        _client = matcher.create(_uri);
        return;
      }
    }

    throw Exception('Unsupported scheme: ${_uri.scheme}');
  }

  static ClientFactory of(BuildContext context) =>
      Provider.of<ClientFactory>(context, listen: false);
}

class _ClientMatcher {
  const _ClientMatcher({required this.match, required this.create});

  final bool Function(Uri uri) match;
  final CusymintClient Function(Uri uri) create;
}

extension on CusymintClient {
  CusymintClient of(BuildContext context) {
    return Provider.of<ClientFactory>(context, listen: false).client;
  }
}
