import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:flutter/material.dart';

class CuTheme {
  static ThemeData of(BuildContext context) {
    final colors = CuColors.of(context);

    return Theme.of(context).copyWith(
      colorScheme: ColorScheme.fromSwatch(
        primarySwatch: colors.materialMint,
        primaryColorDark: colors.mintDark,
        accentColor: colors.mintDark,
        brightness: Brightness.light,
        backgroundColor: colors.mint,
        cardColor: colors.mintLight,
        errorColor: colors.errorColor,
      ),
      scaffoldBackgroundColor: colors.mint,
      backgroundColor: colors.mint,
      textSelectionTheme: TextSelectionThemeData(
        cursorColor: colors.mintDark,
        selectionColor: colors.mintGray,
        selectionHandleColor: colors.mintDark,
      ),
    );
  }
}
