import 'package:cusymint_assets/cusymint_assets.dart';
import 'package:cusymint_ui/cusymint_ui.dart';

class CuCopyTexIconButton extends StatelessWidget {
  const CuCopyTexIconButton({super.key, required this.onPressed});

  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) {
    return IconButton(
      onPressed: onPressed,
      icon: CuAssets.svg.copyTex.svg(
        color: const Color(0xFF222222),
      ),
    );
  }
}
