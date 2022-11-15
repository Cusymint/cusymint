import 'package:cusymint_app/features/client/client_factory.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

class ClientUrlCubit extends Cubit<ClientUrlState> {
  ClientUrlCubit({
    required this.clientFactory,
  }) : super(ClientUrlState.valid(clientFactory.uri.toString()));

  final ClientFactory clientFactory;

  void onChangedUrl(String url) {
    if (clientFactory.isUrlCorrect(url)) {
      emit(ClientUrlState.valid(url));
    } else {
      emit(ClientUrlState.invalid(url));
    }
  }

  void setUrl() {
    if (state.isValid) {
      clientFactory.setUrl(state.url);
    }
  }
}

class ClientUrlState {
  ClientUrlState({required this.url, required this.isValid});

  ClientUrlState.valid(String url) : this(url: url, isValid: true);
  ClientUrlState.invalid(String url) : this(url: url, isValid: false);

  final String url;
  final bool isValid;
}
