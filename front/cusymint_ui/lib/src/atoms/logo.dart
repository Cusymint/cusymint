import 'package:cusymint_assets/cusymint_assets.dart';
import 'package:flutter/material.dart';

class CuLogo extends StatelessWidget {
  const CuLogo({super.key, this.color});

  final Color? color;

  @override
  Widget build(BuildContext context) {
    return CuAssets.svg.logoWide.svg(
      semanticsLabel: 'Cusymint logo',
      color: color,
    );
  }
}
