import 'package:cusymint_assets/cusymint_assets.dart';
import 'package:cusymint_ui/cusymint_ui.dart';

class CuTheme {
  static ThemeData of(BuildContext context) {
    GoogleFonts.config.allowRuntimeFetching = false;

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
      textTheme: GoogleFonts.mavenProTextTheme(),
    );
  }
}
