import 'package:cusymint_assets/cusymint_assets.dart';
import 'package:flutter/material.dart';

class CuLogo extends StatelessWidget {
  const CuLogo({super.key, this.color, this.width});

  final Color? color;
  final double? width;

  @override
  Widget build(BuildContext context) {
    return CuAssets.svg.logoWide.svg(
      width: width,
      semanticsLabel: 'Cusymint logo',
      color: color,
    );
  }
}
