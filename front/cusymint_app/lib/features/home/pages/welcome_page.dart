import 'package:auto_route/auto_route.dart';
import 'package:cusymint_app/features/navigation/navigation.dart';
import 'package:cusymint_ui/cusymint_ui.dart';

class WelcomePage extends StatelessWidget {
  const WelcomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return CuWelcomePageTemplate(
      drawer: WiredDrawer(context: context),
      onTextFieldTap: () {
        context.router.push(HomeRoute(isTextSelected: true));
      },
    );
  }
}
