import 'package:cusymint_app/features/navigation/navigation.dart';
import 'package:cusymint_ui/cusymint_ui.dart';

class SettingsPage extends StatelessWidget {
  const SettingsPage({super.key});

  @override
  Widget build(BuildContext context) {
    return SettingsPageTemplate(
      drawer: WiredDrawer(context: context),
      chosenLanguage: 'English',
      ipAddress: '',
      onIpAddressTap: () {},
      onLanguageTap: () {},
      onLicensesTap: () {
        showLicensePage(
          context: context,
          applicationIcon: CuLogo(
            color: CuColors.of(context).black,
          ),
          applicationName: '',
        );
      },
    );
  }
}
