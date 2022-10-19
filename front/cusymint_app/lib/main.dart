import 'package:cusymint_app/services_provider.dart';
import 'package:cusymint_l10n/cusymint_l10n.dart';
import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:flutter/material.dart';

import 'features/navigation/app_router.gr.dart';

void main() async {
  await CuL10n.ensureInitialized();

  runApp(CuL10n(child: CusymintApp()));
}

class CusymintApp extends StatelessWidget {
  CusymintApp({super.key});

  final _appRouter = AppRouter();

  @override
  Widget build(BuildContext context) {
    return ServicesProvider(
      child: MaterialApp.router(
        locale: context.locale,
        theme: CuTheme.of(context),
        supportedLocales: context.supportedLocales,
        localizationsDelegates: context.localizationDelegates,
        title: 'cusymint',
        routerDelegate: _appRouter.delegate(),
        routeInformationParser: _appRouter.defaultRouteParser(),
      ),
    );
  }
}
