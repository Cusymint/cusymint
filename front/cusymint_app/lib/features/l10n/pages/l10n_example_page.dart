import 'package:cusymint_l10n/cusymint_l10n.dart';
import 'package:flutter/material.dart';

class L10nExamplePage extends StatelessWidget {
  const L10nExamplePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          children: [
            Text('enterIntegral'.tr()),
            Text('tryExamples'.tr()),
            Text('foundResult'.tr(namedArgs: {'timeInMs': '100'})),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () async {
          const locales = CuL10n.supportedLocales;
          final currentLocale = context.locale;
          final nextLocale =
              locales[(locales.indexOf(currentLocale) + 1) % locales.length];

          await context.setLocale(nextLocale);
        },
        child: const Icon(Icons.language),
      ),
    );
  }
}
