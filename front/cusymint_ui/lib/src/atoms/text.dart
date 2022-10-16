import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:flutter/material.dart';

class CuText extends StatelessWidget {
  const CuText(
    this.data, {
    super.key,
    this.color,
    this.fontWeight = FontWeight.normal,
    this.fontSize = 14.0,
  });

  const CuText.med14(
    this.data, {
    super.key,
    this.color,
  })  : fontWeight = FontWeight.normal,
        fontSize = 14.0;

  const CuText.bold14(
    this.data, {
    super.key,
    this.color,
  })  : fontWeight = FontWeight.bold,
        fontSize = 14.0;

  const CuText.med24(
    this.data, {
    super.key,
    this.color,
  })  : fontWeight = FontWeight.normal,
        fontSize = 24.0;

  final String data;
  final CuColor? color;
  final FontWeight fontWeight;
  final double fontSize;

  @override
  Widget build(BuildContext context) {
    final textColor = color ?? CuColors.of(context).textBlack;

    return Text(
      data,
      style: TextStyle(
        color: textColor,
        fontWeight: fontWeight,
        fontSize: fontSize,
      ),
    );
  }
}
