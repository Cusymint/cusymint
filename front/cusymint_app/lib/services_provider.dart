import 'package:cusymint_app/features/client/client_factory.dart';
import 'package:flutter/widgets.dart';
import 'package:provider/provider.dart';

class ServicesProvider extends StatelessWidget {
  const ServicesProvider({super.key, required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        // TODO: add SharedPreferences sync
        Provider<ClientFactory>(
          create: (context) => ClientFactory(),
        ),
      ],
      child: child,
    );
  }
}
