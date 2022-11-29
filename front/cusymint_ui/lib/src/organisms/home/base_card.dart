import 'package:cusymint_ui/cusymint_ui.dart';

class BaseCard extends StatelessWidget {
  const BaseCard({
    super.key,
    this.title,
    this.children = const [],
  });

  final Widget? title;
  final List<Widget> children;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        if (title != null) title!,
        Center(
          child: CuCard(
            child: Column(
              children: children,
            ),
          ),
        )
      ],
    );
  }
}
