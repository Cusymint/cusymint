import 'package:flutter/material.dart';

class CuColor extends Color {
  CuColor(super.value);
}

class CuColors {
  CuColors({
    required this.mint,
    required this.mintLight,
    required this.mintDark,
    required this.mintHeavyish,
    required this.mintGray,
    required this.errorColor,
    required this.textWhite,
    required this.textBlack,
    required this.white,
    required this.black,
  });

  factory CuColors.light() {
    return CuColors(
      mint: CuColor(0xFFB9DED2),
      mintLight: CuColor(0xFFFCFFFE),
      mintDark: CuColor(0xFF002F20),
      mintHeavyish: CuColor(0xFF00744A),
      mintGray: CuColor(0xFF49816F),
      errorColor: CuColor(0xFF9D4529),
      textWhite: CuColor(0xFFFFFFFF),
      textBlack: CuColor(0xFF000000),
      white: CuColor(0xFFFFFFFF),
      black: CuColor(0xFF000000),
    );
  }

  static CuColors of(BuildContext context) {
    // TODO: implement using InheritedWidget
    return CuColors.light();
  }

  final CuColor white;
  final CuColor black;

  final CuColor textWhite;
  final CuColor textBlack;

  final CuColor mint;
  final CuColor mintLight;
  final CuColor mintDark;
  final CuColor mintHeavyish;
  final CuColor mintGray;

  final CuColor errorColor;
}
