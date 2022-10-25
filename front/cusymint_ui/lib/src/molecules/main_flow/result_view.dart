import 'package:cusymint_l10n/cusymint_l10n.dart';
import 'package:cusymint_ui/cusymint_ui.dart';

class CuResultView extends StatefulWidget {
  const CuResultView({
    super.key,
    required this.child,
    required this.shareUtf,
    required this.copyTexToClipboard,
    required this.copyUtfToClipboard,
    required this.duration,
  });

  final Widget child;
  final Duration duration;
  final VoidCallback shareUtf;
  final VoidCallback copyTexToClipboard;
  final VoidCallback copyUtfToClipboard;

  @override
  State<CuResultView> createState() => _CuResultViewState();
}

class _CuResultViewState extends State<CuResultView> {
  final _scrollController = ScrollController();

  @override
  void dispose() {
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        CuText.med14(Strings.foundResult.tr(namedArgs: {
          'timeInMs': widget.duration.inMilliseconds.toString(),
        })),
        CuCard(
          child: Column(
            children: [
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 8.0),
                child: Scrollbar(
                  thumbVisibility: true,
                  controller: _scrollController,
                  child: SingleChildScrollView(
                    scrollDirection: Axis.horizontal,
                    controller: _scrollController,
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: widget.child,
                    ),
                  ),
                ),
              ),
              ButtonBar(
                alignment: MainAxisAlignment.end,
                children: [
                  IconButton(
                    onPressed: widget.shareUtf,
                    // TODO: replace with cusymint icons
                    icon: const Icon(Icons.share),
                  ),
                  IconButton(
                    onPressed: widget.copyTexToClipboard,
                    // TODO: replace with cusymint icons
                    icon: const Icon(Icons.copy),
                  ),
                  IconButton(
                    onPressed: widget.copyUtfToClipboard,
                    // TODO: replace with cusymint icons
                    icon: const Icon(Icons.copy_sharp),
                  ),
                ],
              )
            ],
          ),
        ),
      ],
    );
  }
}
