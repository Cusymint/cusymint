import 'package:auto_route/auto_route.dart';
import 'package:cusymint_ui/cusymint_ui.dart';

import '../navigation.dart';

class WiredDrawer extends CuDrawer {
  WiredDrawer({
    super.key,
    required this.context,
  }) : super(
          onAboutPressed: () {
            context.navigateTo(AboutRoute());
          },
          onHomePressed: () {
            context.router.popUntilRoot();
          },
          onSettingsPressed: () {
            context.navigateTo(const SettingsRoute());
          },
        );

  final BuildContext context;
}
