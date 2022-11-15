import 'package:cusymint_ui/cusymint_ui.dart';

class CuTextButton extends StatelessWidget {
  const CuTextButton({
    super.key,
    required this.text,
    required this.onPressed,
  });

  final String text;
  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) {
    final colors = CuColors.of(context);

    return TextButton(
      onPressed: onPressed,
      child: CuText.bold14(text, color: colors.mintDark),
    );
  }
}
