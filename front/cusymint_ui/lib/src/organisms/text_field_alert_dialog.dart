import 'package:cusymint_ui/cusymint_ui.dart';

class CuTextFieldAlertDialog extends StatelessWidget {
  const CuTextFieldAlertDialog({
    super.key,
    required this.title,
    this.textField,
    required this.onOkPressed,
    required this.onCancelPressed,
  });

  final String title;
  final CuTextField? textField;
  final VoidCallback? onOkPressed;
  final VoidCallback onCancelPressed;

  @override
  Widget build(BuildContext context) {
    return CuAlertDialog(
      title: CuText.med24(title),
      content: textField,
      actions: [
        CuTextButton(
          text: 'Cancel',
          onPressed: onCancelPressed,
        ),
        CuElevatedButton(
          text: 'Ok',
          onPressed: onOkPressed,
        ),
      ],
    );
  }
}
