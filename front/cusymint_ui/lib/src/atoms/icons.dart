import 'package:cusymint_assets/cusymint_assets.dart';
import 'package:cusymint_ui/cusymint_ui.dart';

enum CuIcons {
  none,
  copy,
  copyTex,
  share,
}

class CuIcon extends StatelessWidget {
  const CuIcon(
    this.icon, {
    super.key,
    this.color,
  });

  final CuIcons icon;
  final CuColor? color;

  @override
  Widget build(BuildContext context) {
    switch (icon) {
      case CuIcons.none:
        return const Icon(null);
      case CuIcons.copy:
        return Icon(Icons.copy, color: color);
      case CuIcons.copyTex:
        return CuAssets.icons.copyTex.svg(color: color);
      case CuIcons.share:
        return Icon(Icons.share, color: color);
    }
  }
}
