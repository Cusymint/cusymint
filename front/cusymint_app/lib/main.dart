import 'package:cusymint_app/features/home/pages/home_page.dart';
import 'package:cusymint_l10n/cusymint_l10n.dart';
import 'package:flutter/material.dart';

void main() async {
  await CuL10n.ensureInitialized();

  runApp(const CuL10n(child: CusymintApp()));
}

class CusymintApp extends StatelessWidget {
  const CusymintApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      locale: context.locale,
      supportedLocales: context.supportedLocales,
      localizationsDelegates: context.localizationDelegates,
      title: 'cusymint',
      home: const HomePage(),
    );
  }
}
