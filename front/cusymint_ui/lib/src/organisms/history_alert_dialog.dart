import 'package:cusymint_l10n/cusymint_l10n.dart';
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
      title: CuText.med24(Strings.history.tr()),
      content: Container(
        constraints: const BoxConstraints(maxWidth: 400),
        width: double.maxFinite,
        child: historyItems.isNotEmpty
            ? ListView.builder(
                shrinkWrap: true,
                reverse: true,
                itemCount: historyItems.length,
                itemBuilder: (context, index) {
                  return historyItems[index];
                },
              )
            : CuText.med14(Strings.historyEmpty.tr()),
      ),
      actions: [
        if (historyItems.isNotEmpty)
          CuTextButton(
            text: Strings.clearHistory.tr(),
            onPressed: onClearHistoryPressed,
          )
      ],
    );
  }
}
