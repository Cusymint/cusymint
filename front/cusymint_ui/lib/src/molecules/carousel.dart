import 'package:flutter/material.dart';

class CuCarousel extends StatelessWidget {
  const CuCarousel({
    super.key,
    required this.children,
  });

  final List<Widget> children;

  @override
  Widget build(BuildContext context) {
    return ListView(
      scrollDirection: Axis.horizontal,
      children: children,
    );
  }
}
