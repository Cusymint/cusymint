import 'package:cusymint_ui/cusymint_ui.dart';

class CuScrollableHorizontalWrapper extends StatefulWidget {
  const CuScrollableHorizontalWrapper({
    super.key,
    required this.child,
    this.thumbVisibility = true,
  });

  final bool thumbVisibility;
  final Widget child;

  @override
  State<CuScrollableHorizontalWrapper> createState() =>
      _CuScrollableHorizontalWrapperState();
}

class _CuScrollableHorizontalWrapperState
    extends State<CuScrollableHorizontalWrapper> {
  final _scrollController = ScrollController();

  @override
  void dispose() {
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scrollbar(
      thumbVisibility: widget.thumbVisibility,
      controller: _scrollController,
      child: SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        controller: _scrollController,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: widget.child,
        ),
      ),
    );
  }
}
