import 'package:cusymint_l10n/cusymint_l10n.dart';
import 'package:cusymint_ui/cusymint_ui.dart';

class SettingsPageTemplate<TLocale> extends StatelessWidget {
  const SettingsPageTemplate({
    super.key,
    required this.chosenLanguage,
    required this.ipAddress,
    required this.onIpAddressTap,
    required this.onLicensesTap,
    required this.languageMenuItems,
    this.selectedLocale,
  });

  final String chosenLanguage;
  final String ipAddress;
  final VoidCallback onIpAddressTap;
  final VoidCallback onLicensesTap;
  final List<PopupMenuItem<TLocale>> languageMenuItems;
  final TLocale? selectedLocale;

  @override
  Widget build(BuildContext context) {
    return CuScaffold(
      appBar: CuAppBar(),
      body: Center(
        child: CuSettingsList(
          settingTiles: [
            PopupMenuButton<TLocale>(
              initialValue: selectedLocale,
              padding: EdgeInsets.zero,
              itemBuilder: (context) => languageMenuItems,
              offset: const Offset(1, 0),
              child: CuSettingTile(
                title: CuText(Strings.language.tr()),
                trailing: CuText(chosenLanguage),
              ),
            ),
            CuSettingTile(
              title: CuText(Strings.ipAddress.tr()),
              trailing: CuText(ipAddress),
              onTap: onIpAddressTap,
            ),
            CuSettingTile(
              title: CuText(Strings.licenses.tr()),
              onTap: onLicensesTap,
            ),
          ],
        ),
      ),
    );
  }
}
