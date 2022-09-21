import 'package:easy_localization/easy_localization.dart';
import 'package:flutter/widgets.dart';

class CuL10n extends StatelessWidget {
  const CuL10n({super.key, required this.child});

  final Widget child;

  static Future ensureInitialized() async {
    WidgetsFlutterBinding.ensureInitialized();
    await EasyLocalization.ensureInitialized();
  }

  static const supportedLocales = [
    Locale('pl'),
    Locale('en'),
  ];

  static const fallbackLocale = Locale('en');

  static const path = 'packages/cusymint_l10n/assets/translations';

  @override
  Widget build(BuildContext context) {
    return EasyLocalization(
      supportedLocales: supportedLocales,
      fallbackLocale: fallbackLocale,
      path: path,
      child: child,
    );
  }
}
