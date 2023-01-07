import 'package:cusymint_l10n/cusymint_l10n.dart';
import 'package:cusymint_ui/cusymint_ui.dart';

class CuButtonRowCallbacks {
  const CuButtonRowCallbacks({
    required this.onCopyTex,
    required this.onCopyUtf,
    required this.onShareUtf,
    required this.onStepsTap,
  });

  final VoidCallback onCopyTex;
  final VoidCallback onCopyUtf;
  final VoidCallback onShareUtf;
  final VoidCallback onStepsTap;
}

class CuAnimatedHomeCard extends StatefulWidget {
  const CuAnimatedHomeCard({
    super.key,
    this.title,
    this.hasCriticalErrors = false,
    this.errors = const [],
    this.inputInTex,
    this.outputInTex,
    this.isLoading = false,
    this.buttonRowCallbacks,
    this.steps,
  });

  final String? title;

  final List<String> errors;
  final bool hasCriticalErrors;

  final String? inputInTex;
  final String? outputInTex;
  final String? steps;
  final bool isLoading;

  final CuButtonRowCallbacks? buttonRowCallbacks;

  final Duration duration = const Duration(milliseconds: 200);

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
              AnimatedSize(
                duration: widget.duration,
                child: _HomeStepsView(
                  steps: widget.steps,
                ),
              ),
              AnimatedSize(
                duration: widget.duration,
                child: _HomeButtonRow(
                  callbacks: widget.buttonRowCallbacks,
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
        child: AnimatedTexView(tex, colors: colors, isLoading: isLoading),
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

class _HomeStepsView extends StatelessWidget {
  const _HomeStepsView({this.steps});

  final String? steps;

  @override
  Widget build(BuildContext context) {
    if (steps != null) {
      final colors = CuColors.of(context);

      return CuScrollableHorizontalWrapper(
        child: TexView(steps!, color: colors.black),
      );
    }

    return const SizedBox.shrink();
  }
}

class AnimatedTexView extends StatefulWidget {
  const AnimatedTexView(
    this.data, {
    super.key,
    this.duration = const Duration(milliseconds: 700),
    this.isLoading = false,
    required this.colors,
  });

  final Duration duration;
  final String data;
  final bool isLoading;

  final CuColors colors;

  @override
  State<AnimatedTexView> createState() => _AnimatedTexViewState();
}

class _AnimatedTexViewState extends State<AnimatedTexView>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  late final Animation<Color?> _colorAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: widget.duration,
    )..repeat(
        reverse: true,
      );

    _colorAnimation = ColorTween(
      begin: widget.colors.black,
      end: widget.colors.grayDark,
    ).animate(_controller);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final animated = AnimatedBuilder(
      animation: _controller,
      builder: (context, child) => TexView(
        widget.data,
        color: _colorAnimation.value,
      ),
    );

    final stale = TexView(widget.data);

    final child = widget.isLoading ? animated : stale;

    return child;
  }
}

class _HomeButtonRow extends StatelessWidget {
  const _HomeButtonRow({required this.callbacks});

  final CuButtonRowCallbacks? callbacks;

  @override
  Widget build(BuildContext context) {
    if (callbacks == null) {
      return const SizedBox.shrink();
    }

    return ButtonBar(
      alignment: MainAxisAlignment.center,
      mainAxisSize: MainAxisSize.min,
      children: [
        IconButton(
          onPressed: callbacks!.onShareUtf,
          icon: const Icon(Icons.share),
        ),
        CuCopyTexIconButton(
          onPressed: callbacks!.onCopyTex,
        ),
        IconButton(
          onPressed: callbacks!.onCopyUtf,
          icon: const Icon(Icons.copy),
        ),
        CuElevatedButton(
          text: Strings.steps.tr(),
          onPressed: callbacks!.onStepsTap,
        ),
      ],
    );
  }
}
