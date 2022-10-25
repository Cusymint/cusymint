import 'package:flutter/material.dart';

class CuColor extends Color {
  CuColor(super.value);
}

class CuColors {
  CuColors({
    required this.mint,
    required this.materialMint,
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
      materialMint: MaterialColor(
        0xFF002F20,
        <int, Color>{
          50: CuColor(0xFFE2F2ED),
          100: CuColor(0xFFB9DED2),
          200: CuColor(0xFF8dcab6),
          300: CuColor(0xFF64b49a),
          400: CuColor(0xFF4aa487),
          500: CuColor(0xFF3b9475),
          600: CuColor(0xFF368769),
          700: CuColor(0xFF30775B),
          800: CuColor(0xFF29674E),
          900: CuColor(0xFF1e4b35),
        },
      ),
    );
  }

  /// The main color palette of the app.
  ///
  /// Use instead of directly using constructors in case
  /// of a need to add dark mode support.
  static CuColors of(BuildContext context) {
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
  final MaterialColor materialMint;

  final CuColor errorColor;
}
