import 'package:cusymint_client_mock/cusymint_client_mock.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:provider/provider.dart';

import 'features/home/blocs/client_cubit.dart';

class ServicesProvider extends StatelessWidget {
  const ServicesProvider({super.key, required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        Provider<CusymintClient>(
          // TODO: replace with real client
          create: (context) => CusymintClientMock(
            fakeResponse: ResponseMockFactory.defaultResponse,
          ),
        ),
        BlocProvider(
          create: (context) => ClientCubit(
            client: Provider.of<CusymintClient>(context, listen: false),
          ),
        ),
      ],
      child: child,
    );
  }
}
