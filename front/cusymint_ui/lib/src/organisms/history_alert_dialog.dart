import 'package:cusymint_ui/cusymint_ui.dart';

class CuHistoryAlertDialog extends StatelessWidget {
  const CuHistoryAlertDialog({
    super.key,
    required this.onClearHistoryPressed,
    required this.historyItems,
  });

  final List<CuHistoryItem> historyItems;
  final VoidCallback onClearHistoryPressed;

  @override
  Widget build(BuildContext context) {
    return CuAlertDialog(
      // TODO: add to strings
      title: const CuText.med24('History'),
      content: ListView(
        shrinkWrap: true,
        reverse: true,
        children: historyItems,
      ),
      actions: [
        CuTextButton(
          text: 'Clear history',
          onPressed: onClearHistoryPressed,
        )
      ],
    );
  }
}
