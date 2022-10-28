import 'package:cusymint_l10n/cusymint_l10n.dart';
import 'package:cusymint_ui/cusymint_ui.dart';

class SettingsPageTemplate extends StatelessWidget {
  const SettingsPageTemplate({
    super.key,
    required this.drawer,
    required this.chosenLanguage,
    required this.ipAddress,
    required this.onLanguageTap,
    required this.onIpAddressTap,
    required this.onLicensesTap,
  });

  final CuDrawer drawer;
  final String chosenLanguage;
  final String ipAddress;
  final VoidCallback onLanguageTap;
  final VoidCallback onIpAddressTap;
  final VoidCallback onLicensesTap;

  @override
  Widget build(BuildContext context) {
    return CuScaffold(
      appBar: CuAppBar(),
      drawer: drawer,
      body: Center(
        child: CuSettingsList(
          settingTiles: [
            CuSettingTile(
              title: CuText(Strings.language.tr()),
              trailing: CuText(chosenLanguage),
              onTap: onLanguageTap,
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
