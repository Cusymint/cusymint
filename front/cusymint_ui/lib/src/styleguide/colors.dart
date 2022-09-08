import 'package:flutter/material.dart';

class CuColor extends Color {
  CuColor(super.value);
}

class CuColors {
  CuColors({
    required this.textWhite,
    required this.textBlack,
    required this.backgroundHighlight,
    required this.background,
    required this.white,
    required this.black,
  });

  final CuColor white;
  final CuColor black;

  final CuColor textWhite;
  final CuColor textBlack;

  final CuColor backgroundHighlight;
  final CuColor background;
}
