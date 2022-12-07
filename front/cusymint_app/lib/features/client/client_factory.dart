import 'package:cusymint_client/cusymint_client.dart';
import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

class ClientFactory {
  ClientFactory({this.sharedPreferences}) {
    initialize(false);
  }

  ClientFactory.withStorage() {
    initialize(true);
  }

  SharedPreferences? sharedPreferences;

  Uri get uri => _uri;
  Uri _uri = Uri.parse(_defaultUri);

  CusymintClient get client => _client;
  CusymintClient _client = const CusymintClientMock();

  static const _defaultUri = 'ws://localhost:8000/websocket';
  static const _clientUrlKey = 'clientUrl';

  final _clientMatchers = <_ClientMatcher>[
    _ClientMatcher(
      match: (uri) => uri.host == '' && uri.path == 'mock',
      create: (_) => const CusymintClientMock(
        fakeResponse: ResponseMockFactory.defaultResponse,
      ),
    ),
    _ClientMatcher(
      match: (uri) => uri.host == '' && uri.path == 'errors',
      create: (_) => const CusymintClientMock(
        fakeResponse: ResponseMockFactory.validationErrors,
      ),
    ),
    _ClientMatcher(
      match: (uri) => uri.scheme == 'ws',
      create: (uri) => CusymintClientJsonRpc(uri: uri),
    ),
  ];

  void initialize(bool initializeStorage) async {
    if (initializeStorage) {
      sharedPreferences = await SharedPreferences.getInstance();
    }
    final storedUrl = _readUrlFromStorage();
    if (storedUrl != null) {
      setUrl(storedUrl);
    }
  }

  bool isUrlCorrect(String url) {
    for (final matcher in _clientMatchers) {
      if (matcher.match(Uri.parse(url))) {
        return true;
      }
    }

    return false;
  }

  Future<void> setUrl(String url) async {
    _uri = Uri.parse(url);

    for (final matcher in _clientMatchers) {
      if (matcher.match(_uri)) {
        _client = matcher.create(_uri);
        await sharedPreferences?.setString(_clientUrlKey, url);
        return;
      }
    }

    throw Exception('Unsupported scheme: ${_uri.scheme}');
  }

  String? _readUrlFromStorage() => sharedPreferences?.getString(_clientUrlKey);

  static ClientFactory of(BuildContext context) =>
      Provider.of<ClientFactory>(context, listen: false);
}

class _ClientMatcher {
  const _ClientMatcher({required this.match, required this.create});

  final bool Function(Uri uri) match;
  final CusymintClient Function(Uri uri) create;
}
