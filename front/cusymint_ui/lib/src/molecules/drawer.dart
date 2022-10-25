import 'package:cusymint_l10n/cusymint_l10n.dart';
import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:flutter/material.dart';

class CuDrawer extends StatelessWidget {
  const CuDrawer({
    super.key,
    required this.onHomePressed,
    required this.onSettingsPressed,
    required this.onAboutPressed,
  });

  final VoidCallback onHomePressed;
  final VoidCallback onSettingsPressed;
  final VoidCallback onAboutPressed;

  @override
  Widget build(BuildContext context) {
    final colors = CuColors.of(context);

    return Drawer(
      child: DecoratedBox(
        decoration: CuBlackBoardDecoration(),
        child: ListView(
          padding: EdgeInsets.zero,
          children: [
            DrawerHeader(child: CuLogo(color: colors.white)),
            ListTile(
              title: CuText.med24(Strings.home.tr(), color: colors.textWhite),
              onTap: onHomePressed,
            ),
            ListTile(
              title:
                  CuText.med24(Strings.settings.tr(), color: colors.textWhite),
              onTap: onSettingsPressed,
            ),
            ListTile(
              title: CuText.med24(Strings.about.tr(), color: colors.textWhite),
              onTap: onAboutPressed,
            ),
          ],
        ),
      ),
    );
  }
}
