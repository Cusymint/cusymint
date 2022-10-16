import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:flutter/material.dart';

class CuCard extends StatelessWidget {
  const CuCard({super.key, this.child});

  final Widget? child;

  @override
  Widget build(BuildContext context) {
    final colors = CuColors.of(context);

    return Card(
      color: colors.mintLight,
      child: child,
    );
  }
}
