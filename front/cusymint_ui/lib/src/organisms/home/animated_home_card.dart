import 'package:cusymint_ui/cusymint_ui.dart';

class CuAnimatedHomeCard extends StatefulWidget {
  const CuAnimatedHomeCard({
    super.key,
    this.title,
    this.hasCriticalErrors = false,
    this.errors = const [],
    this.inputInTex,
    this.outputInTex,
    this.isLoading = false,
    this.onCopyUtf,
    this.onCopyTex,
    this.onShareUtf,
  });

  final String? title;

  final List<String> errors;
  final bool hasCriticalErrors;

  final String? inputInTex;
  final String? outputInTex;
  final bool isLoading;

  final VoidCallback? onCopyUtf;
  final VoidCallback? onCopyTex;
  final VoidCallback? onShareUtf;

  final Duration duration = const Duration(milliseconds: 200);

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
        if (widget.title != null)
          AnimatedSize(
            duration: widget.duration,
            child: CuText.med14(widget.title!),
          ),
        AnimatedContainer(
          duration: widget.duration,
          decoration: BoxDecoration(
            color: widget.hasCriticalErrors ? colors.errorColor : colors.white,
            borderRadius: BorderRadius.circular(8),
            boxShadow: [
              BoxShadow(
                color: colors.black.withOpacity(0.4),
                blurRadius: 8,
                offset: const Offset(0, 4),
              ),
            ],
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              AnimatedSize(
                duration: widget.duration,
                child: _ErrorListView(
                  errors: widget.errors,
                  hasBottomWidget: widget.inputInTex != null,
                ),
              ),
              AnimatedSize(
                duration: widget.duration,
                child: _HomeTexView(
                  inputInTex: widget.inputInTex,
                  outputInTex: widget.outputInTex,
                  isLoading: widget.isLoading,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _ErrorListView extends StatelessWidget {
  const _ErrorListView({
    required this.errors,
    required this.hasBottomWidget,
  });

  final List<String> errors;
  final bool hasBottomWidget;

  @override
  Widget build(BuildContext context) {
    if (errors.isEmpty) {
      return const SizedBox.shrink();
    }

    final padding = hasBottomWidget
        ? const EdgeInsets.fromLTRB(8.0, 8.0, 8.0, 0.0)
        : const EdgeInsets.all(8.0);

    return Padding(
      padding: padding,
      child: Column(
        children: [
          for (final error in errors) CuText.med14(error),
        ],
      ),
    );
  }
}

class _HomeTexView extends StatelessWidget {
  const _HomeTexView({
    required this.inputInTex,
    required this.outputInTex,
    required this.isLoading,
  });

  final String? inputInTex;
  final String? outputInTex;
  final bool isLoading;

  @override
  Widget build(BuildContext context) {
    if (inputInTex != null) {
      final tex = _formatTex();
      final colors = CuColors.of(context);

      return CuScrollableHorizontalWrapper(
        child: TexView(tex),
      );
    }

    return const SizedBox.shrink();
  }

  String _formatTex() {
    if (inputInTex != null && outputInTex != null) {
      return '${inputInTex!} = ${outputInTex!}';
    }

    if (inputInTex != null) {
      return inputInTex!;
    }

    return '';
  }
}
