import 'package:cusymint_ui/cusymint_ui.dart';

class CuAppBar extends AppBar {
  CuAppBar({
    this.onMenuPressed,
    this.hasLogo = true,
    super.key,
  }) : super(
          // TODO: Fix sizing, logo is larger than IconButton
          title: hasLogo ? const Hero(tag: 'logo', child: CuLogo()) : null,
          centerTitle: true,
          actions: [
            if (onMenuPressed != null)
              IconButton(
                icon: const Icon(Icons.menu),
                onPressed: onMenuPressed,
              ),
          ],
          elevation: 0,
          backgroundColor: Colors.transparent,
        );

  final VoidCallback? onMenuPressed;
  final bool hasLogo;
}
