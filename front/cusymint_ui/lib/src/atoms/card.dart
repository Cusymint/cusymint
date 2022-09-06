import 'package:flutter/material.dart';

class CuCard extends StatelessWidget {
  const CuCard({super.key, this.child});

  final Widget? child;

  @override
  Widget build(BuildContext context) {
    // TODO: apply app style
    return Card(
      child: child,
    );
  }
}
