import 'package:cusymint_app/features/client/client_factory.dart';
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
        Provider<ClientFactory>(
          create: (context) => ClientFactory(),
        ),
        BlocProvider(
          create: (context) => ClientCubit(
            clientFactory: ClientFactory.of(context),
          ),
        ),
      ],
      child: child,
    );
  }
}
