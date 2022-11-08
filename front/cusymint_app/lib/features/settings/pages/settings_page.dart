import 'package:cusymint_app/features/navigation/navigation.dart';
import 'package:cusymint_l10n/cusymint_l10n.dart';
import 'package:cusymint_ui/cusymint_ui.dart';

class SettingsPage extends StatelessWidget {
  const SettingsPage({super.key});

  @override
  Widget build(BuildContext context) {
    return SettingsPageTemplate<Locale>(
      drawer: WiredDrawer(context: context),
      chosenLanguage: Strings.languageCurrent.tr(),
      ipAddress: '',
      onIpAddressTap: () {},
      onLicensesTap: () {
        showLicensePage(
          context: context,
          applicationIcon: CuLogo(
            color: CuColors.of(context).black,
          ),
          applicationName: '',
        );
      },
      selectedLocale: context.locale,
      languageMenuItems: [
        LocalePopupMenuItem(
          context: context,
          locale: const Locale('en'),
          language: Strings.languageEnglish.tr(),
        ),
        LocalePopupMenuItem(
          context: context,
          locale: const Locale('pl'),
          language: Strings.languagePolish.tr(),
        ),
      ],
    );
  }
}

class LocalePopupMenuItem extends PopupMenuItem<Locale> {
  LocalePopupMenuItem({
    super.key,
    required this.locale,
    required this.context,
    required this.language,
  }) : super(
          value: locale,
          child: CuText.med14(language),
          onTap: () async {
            await context.setLocale(locale);
          },
        );

  final Locale locale;
  final String language;
  final BuildContext context;
}
