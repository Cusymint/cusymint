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
      content: Container(
        constraints: const BoxConstraints(maxWidth: 400),
        width: double.maxFinite,
        child: ListView.builder(
          shrinkWrap: true,
          reverse: true,
          itemCount: historyItems.length,
          itemBuilder: (context, index) {
            return historyItems[index];
          },
        ),
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
