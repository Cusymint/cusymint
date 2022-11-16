import 'package:auto_route/auto_route.dart';
import 'package:cusymint_app/features/navigation/navigation.dart';
import 'package:cusymint_l10n/cusymint_l10n.dart';
import 'package:cusymint_ui/cusymint_ui.dart';

class WelcomePage extends StatelessWidget {
  const WelcomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return CuWelcomePageTemplate(
      // Note this is fix for the issue with third party packages
      key: Key(context.locale.languageCode),
      drawer: WiredDrawer(context: context),
      onTextFieldTap: () {
        context.router.push(HomeRoute(isTextSelected: true));
      },
    );
  }
}
