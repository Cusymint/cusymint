import 'package:cusymint_ui/cusymint_ui.dart';

class CuAppBar extends AppBar {
  CuAppBar({
    this.hasLogo = true,
    List<Widget> actions = const [],
    super.key,
  }) : super(
          title: hasLogo ? const Hero(tag: 'logo', child: CuLogo()) : null,
          centerTitle: true,
          actions: actions,
          toolbarHeight: 70,
          elevation: 0,
          backgroundColor: Colors.transparent,
        );

  final bool hasLogo;
}
