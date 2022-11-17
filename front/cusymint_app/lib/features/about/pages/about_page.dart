import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:url_launcher/url_launcher.dart' as url_launcher;

class AboutPage extends StatelessWidget {
  AboutPage({super.key});

  static const _githubUrl = 'https://github.com/Cusymint/cusymint';
  final Uri _githubUri = Uri.parse(_githubUrl);

  @override
  Widget build(BuildContext context) {
    return AboutTemplate(
      onGithubTap: () async {
        await url_launcher.launchUrl(_githubUri);
      },
    );
  }
}
