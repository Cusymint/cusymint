import 'package:cusymint_l10n/cusymint_l10n.dart';
import 'package:cusymint_ui/cusymint_ui.dart';

class CuHistoryDialog extends StatelessWidget {
  const CuHistoryDialog({
    super.key,
    required this.onClearHistoryPressed,
    required this.onCancelPressed,
    required this.historyItems,
  });

  final List<CuHistoryItem> historyItems;
  final VoidCallback onClearHistoryPressed;
  final VoidCallback onCancelPressed;

  @override
  Widget build(BuildContext context) {
    return CuAlertDialog(
      title: CuText.med24(Strings.history.tr()),
      content: Container(
        constraints: const BoxConstraints(maxWidth: 400),
        width: double.maxFinite,
        child: AnimatedSize(
          duration: const Duration(milliseconds: 100),
          child: AnimatedSwitcher(
            duration: const Duration(milliseconds: 100),
            child: historyItems.isNotEmpty
                ? ListView.builder(
                    shrinkWrap: true,
                    itemCount: historyItems.length,
                    itemBuilder: (context, index) {
                      return historyItems[index];
                    },
                  )
                : CuText.med14(Strings.historyEmpty.tr()),
          ),
        ),
      ),
      actions: [
        CuTextButton(
          text: Strings.cancel.tr(),
          onPressed: onCancelPressed,
        ),
        if (historyItems.isNotEmpty)
          CuTextButton(
            text: Strings.clearHistory.tr(),
            onPressed: onClearHistoryPressed,
          ),
      ],
    );
  }
}
