import 'package:flutter/material.dart';
import 'package:flutter_math_fork/flutter_math.dart';

/// Wrapper for a third party Tex rendering widget.
///
/// List of supported commands:
/// https://katex.org/docs/supported.html
class TexView extends StatelessWidget {
  const TexView(
    this.data, {
    super.key,
    this.fontScale,
  });

  final String data;
  final double? fontScale;

  @override
  Widget build(BuildContext context) {
    return Math.tex(
      data,
      textScaleFactor: fontScale,
    );
  }
}
