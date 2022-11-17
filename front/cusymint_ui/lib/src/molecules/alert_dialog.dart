import 'package:cusymint_ui/cusymint_ui.dart';

class CuAlertDialog extends StatelessWidget {
  const CuAlertDialog({super.key, this.title, this.content, this.actions});

  final CuText? title;
  final Widget? content;
  final List<Widget>? actions;

  @override
  Widget build(BuildContext context) {
    final colors = CuColors.of(context);

    return AlertDialog(
      title: title,
      content: content,
      actions: actions,
      backgroundColor: colors.mintLight,
    );
  }
}
