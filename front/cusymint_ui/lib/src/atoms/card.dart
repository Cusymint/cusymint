import 'package:cusymint_ui/cusymint_ui.dart';

class CuCard extends StatelessWidget {
  const CuCard({super.key, this.child});

  final Widget? child;

  @override
  Widget build(BuildContext context) {
    final colors = CuColors.of(context);

    return Card(
      elevation: 8,
      color: colors.mintLight,
      child: child,
    );
  }
}
