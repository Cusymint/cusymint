import 'package:auto_route/auto_route.dart';
import 'package:cusymint_ui/cusymint_ui.dart';

import '../navigation.dart';

class WiredDrawer extends CuDrawer {
  WiredDrawer({
    super.key,
    required this.context,
  }) : super(
          onAboutPressed: () async {
            _hideDrawer(context);
            await context.navigateTo(AboutRoute());
          },
          onHomePressed: () async {
            await _hideDrawer(context);
            context.router.popUntilRoot();
          },
          onSettingsPressed: () async {
            _hideDrawer(context);
            await context.navigateTo(const SettingsRoute());
          },
        );

  static Future<void> _hideDrawer(BuildContext context) async {
    await context.router.pop();
  }

  final BuildContext context;
}
