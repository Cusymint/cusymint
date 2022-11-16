import 'package:cusymint_ui/cusymint_ui.dart';

class CuElevatedButton extends StatelessWidget {
  const CuElevatedButton({
    super.key,
    required this.text,
    required this.onPressed,
  });

  final String text;
  final VoidCallback? onPressed;

  @override
  Widget build(BuildContext context) {
    final colors = CuColors.of(context);

    return ElevatedButton(
      onPressed: onPressed,
      child: CuText.bold14(
        text.toUpperCase(),
        color: colors.textWhite,
      ),
    );
  }
}
