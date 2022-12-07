import 'package:cusymint_app/features/client/client_factory.dart';
import 'package:cusymint_app/features/home/blocs/example_integrals_cubit.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:provider/provider.dart';

class ServicesProvider extends StatelessWidget {
  const ServicesProvider({super.key, required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        Provider<ClientFactory>(
          create: (context) => ClientFactory.withStorage(),
        ),
        BlocProvider<ExampleIntegralsCubit>(
          create: (_) => ExampleIntegralsCubit()..generateIntegrals(),
        ),
      ],
      child: child,
    );
  }
}
