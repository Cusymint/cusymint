import 'package:cusymint_ui/cusymint_ui.dart';

class CuAnimatedHomeCard extends StatefulWidget {
  const CuAnimatedHomeCard({
    super.key,
    this.title,
    this.hasCriticalErrors = false,
    this.errors = const [],
    this.inputInTex,
    this.outputInTex,
    this.onCopyUtf,
    this.onCopyTex,
    this.onShareUtf,
  });

  final String? title;

  final List<String> errors;
  final bool hasCriticalErrors;

  final String? inputInTex;
  final String? outputInTex;

  final VoidCallback? onCopyUtf;
  final VoidCallback? onCopyTex;
  final VoidCallback? onShareUtf;

  bool get hasAllCallbacks =>
      onCopyUtf != null && onCopyTex != null && onShareUtf != null;

  @override
  State<CuAnimatedHomeCard> createState() => _CuAnimatedHomeCardState();
}

class _CuAnimatedHomeCardState extends State<CuAnimatedHomeCard> {
  @override
  Widget build(BuildContext context) {
    final colors = CuColors.of(context);

    return Column(
      children: [
        AnimatedSize(
          duration: const Duration(milliseconds: 300),
          child: widget.title != null ? CuText.med14(widget.title!) : null,
        ),
        AnimatedSize(
          duration: const Duration(milliseconds: 100),
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 100),
            curve: Curves.easeInOut,
            decoration: BoxDecoration(
              color:
                  widget.hasCriticalErrors ? colors.errorColor : colors.white,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                if (widget.errors.isNotEmpty)
                  for (final error in widget.errors) CuText.med14(error),
                if (widget.inputInTex != null && widget.outputInTex != null)
                  Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: TexView(
                        '${widget.inputInTex!} = ${widget.outputInTex!}'),
                  ),
                if (widget.inputInTex != null && widget.outputInTex == null)
                  Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: TexView(widget.inputInTex!),
                  ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}
